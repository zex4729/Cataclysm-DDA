#include "render3d.h"

#include <algorithm>
#include <cmath>

namespace render3d
{

namespace
{

constexpr float pi = 3.14159265358979323846f;

float to_radians( float degrees )
{
    return degrees * pi / 180.0f;
}

// Interpolates a quad corner at parameters (s, t) in [0, 1]: s runs from the left edge to the
// right edge, t from the top edge to the bottom edge.
template<typename T>
T bilerp( const T &top_left, const T &top_right, const T &bottom_right, const T &bottom_left,
          float s, float t )
{
    const T top = top_left + ( top_right - top_left ) * s;
    const T bottom = bottom_left + ( bottom_right - bottom_left ) * s;
    return top + ( bottom - top ) * t;
}

float bilerp_scalar( const float *c, float s, float t )
{
    return bilerp( c[0], c[1], c[2], c[3], s, t );
}

view_vertex make_view_vertex( const vec3 corners[4], const float u[4], const float v[4], float s,
                              float t )
{
    view_vertex out;
    out.pos = bilerp( corners[0], corners[1], corners[2], corners[3], s, t );
    out.u = bilerp_scalar( u, s, t );
    out.v = bilerp_scalar( v, s, t );
    return out;
}

// Ratio between the far and the near depth of an edge, always >= 1.
float depth_ratio( float za, float zb, float near_z )
{
    const float lo = std::max( std::min( za, zb ), near_z );
    const float hi = std::max( std::max( za, zb ), near_z );
    return hi / lo;
}

int subdivisions_for_ratio( float ratio, int max_subdivisions )
{
    // Splitting an edge into n parts divides the depth ratio of each part to roughly
    // ratio^(1/n). 1.12 is where the affine texture error stops being noticeable.
    constexpr float tolerated_ratio = 1.12f;
    if( ratio <= tolerated_ratio ) {
        return 1;
    }
    const int n = static_cast<int>( std::ceil( std::log( ratio ) / std::log( tolerated_ratio ) ) );
    return std::clamp( n, 1, std::max( max_subdivisions, 1 ) );
}

} // namespace

camera::camera( const vec3 &position, float yaw_deg, float pitch_deg, float fov_deg,
                int viewport_w, int viewport_h )
    : position_( position ), viewport_w_( viewport_w ), viewport_h_( viewport_h )
{
    const float yaw = to_radians( yaw_deg );
    const float pitch = to_radians( std::clamp( pitch_deg, -89.0f, 89.0f ) );
    const float fx = std::sin( yaw );
    const float fy = -std::cos( yaw );
    const float cp = std::cos( pitch );
    const float sp = std::sin( pitch );
    forward_ = { cp * fx, cp * fy, sp };
    right_ = { std::cos( yaw ), std::sin( yaw ), 0.0f };
    up_ = { -sp * fx, -sp * fy, cp };
    half_tan_fov_ = std::tan( to_radians( std::clamp( fov_deg, 10.0f, 170.0f ) ) * 0.5f );
    focal_ = static_cast<float>( viewport_w_ ) * 0.5f / half_tan_fov_;
}

vec3 camera::to_view( const vec3 &world ) const
{
    const vec3 d = world - position_;
    return { dot( d, right_ ), dot( d, up_ ), dot( d, forward_ ) };
}

void camera::project( const vec3 &view, float &sx, float &sy ) const
{
    const float inv_depth = 1.0f / view.z;
    sx = static_cast<float>( viewport_w_ ) * 0.5f + view.x * inv_depth * focal_;
    sy = static_cast<float>( viewport_h_ ) * 0.5f - view.y * inv_depth * focal_;
}

vec3 camera::pixel_direction( float sx, float sy ) const
{
    const float vx = ( sx - static_cast<float>( viewport_w_ ) * 0.5f ) / focal_;
    const float vy = -( sy - static_cast<float>( viewport_h_ ) * 0.5f ) / focal_;
    const vec3 dir = right_ * vx + up_ * vy + forward_;
    const float len = std::sqrt( dot( dir, dir ) );
    return dir * ( 1.0f / len );
}

void grow_quad( const camera &cam, quad &q, float pixels )
{
    const vec3 middle = bilerp( q.corner[0], q.corner[1], q.corner[2], q.corner[3], 0.5f, 0.5f );
    const vec3 across = q.corner[1] - q.corner[0];
    const vec3 down = q.corner[3] - q.corner[0];
    const vec3 view_middle = cam.to_view( middle );
    if( view_middle.z <= cam.near_plane() ) {
        return;
    }
    float mx = 0.0f;
    float my = 0.0f;
    cam.project( view_middle, mx, my );

    // How many pixels one side of the quad is long on the screen, seen from its middle. Sides
    // that are seen edge on are (nearly) zero pixels long, and would grow without limit.
    const auto pixels_along = [&]( const vec3 & side ) {
        const vec3 view_end = cam.to_view( middle + side * 0.5f );
        if( view_end.z <= cam.near_plane() ) {
            return 0.0f;
        }
        float ex = 0.0f;
        float ey = 0.0f;
        cam.project( view_end, ex, ey );
        return 2.0f * std::hypot( ex - mx, ey - my );
    };
    // Never more than a quarter: this is a patch for rounding, not a way to make things bigger
    constexpr float max_growth = 0.25f;
    const auto growth = [&]( const vec3 & side ) {
        const float length = pixels_along( side );
        return length > pixels / max_growth ? pixels / length : max_growth;
    };
    const float s = growth( across );
    const float t = growth( down );
    const vec3 c0 = q.corner[0];
    const vec3 c1 = q.corner[1];
    const vec3 c2 = q.corner[2];
    const vec3 c3 = q.corner[3];
    q.corner[0] = bilerp( c0, c1, c2, c3, -s, -t );
    q.corner[1] = bilerp( c0, c1, c2, c3, 1.0f + s, -t );
    q.corner[2] = bilerp( c0, c1, c2, c3, 1.0f + s, 1.0f + t );
    q.corner[3] = bilerp( c0, c1, c2, c3, -s, 1.0f + t );
}

int clip_near( const view_vertex *in, int in_count, float near_z, view_vertex *out )
{
    int out_count = 0;
    for( int i = 0; i < in_count; ++i ) {
        const view_vertex &cur = in[i];
        const view_vertex &next = in[( i + 1 ) % in_count];
        const bool cur_inside = cur.pos.z >= near_z;
        const bool next_inside = next.pos.z >= near_z;
        if( cur_inside ) {
            out[out_count++] = cur;
        }
        if( cur_inside != next_inside ) {
            const float t = ( near_z - cur.pos.z ) / ( next.pos.z - cur.pos.z );
            view_vertex cut;
            cut.pos = lerp( cur.pos, next.pos, t );
            // The clip plane is z == near_z, make sure rounding does not leave it behind it
            cut.pos.z = near_z;
            cut.u = cur.u + ( next.u - cur.u ) * t;
            cut.v = cur.v + ( next.v - cur.v ) * t;
            out[out_count++] = cut;
        }
    }
    return out_count;
}

int tessellate( const camera &cam, const quad &q, int nu, int nv, mesh &out )
{
    nu = std::max( nu, 1 );
    nv = std::max( nv, 1 );
    const float near_z = cam.near_plane();

    vec3 view_corners[4];
    for( int i = 0; i < 4; ++i ) {
        view_corners[i] = cam.to_view( q.corner[i] );
    }

    // Vertices of the sub quad grid, in view space. Sub quads that are completely in front of the
    // near plane share their projected corners.
    const int stride = nu + 1;
    std::vector<view_vertex> grid( static_cast<size_t>( stride ) * ( nv + 1 ) );
    std::vector<int64_t> projected( grid.size(), -1 );
    for( int j = 0; j <= nv; ++j ) {
        for( int i = 0; i <= nu; ++i ) {
            grid[static_cast<size_t>( j ) * stride + i] = make_view_vertex(
                        view_corners, q.u, q.v, static_cast<float>( i ) / static_cast<float>( nu ),
                        static_cast<float>( j ) / static_cast<float>( nv ) );
        }
    }

    const auto emit = [&]( const view_vertex & vv ) {
        screen_vertex sv;
        cam.project( vv.pos, sv.x, sv.y );
        sv.u = vv.u;
        sv.v = vv.v;
        sv.depth = vv.pos.z;
        out.vertices.push_back( sv );
        return static_cast<uint32_t>( out.vertices.size() - 1 );
    };
    const auto shared = [&]( size_t grid_index ) {
        if( projected[grid_index] < 0 ) {
            projected[grid_index] = emit( grid[grid_index] );
        }
        return static_cast<uint32_t>( projected[grid_index] );
    };

    int triangles = 0;
    for( int j = 0; j < nv; ++j ) {
        for( int i = 0; i < nu; ++i ) {
            const size_t i00 = static_cast<size_t>( j ) * stride + i;
            const size_t i10 = i00 + 1;
            const size_t i01 = i00 + stride;
            const size_t i11 = i01 + 1;
            const view_vertex sub[4] = { grid[i00], grid[i10], grid[i11], grid[i01] };

            const bool all_in_front = sub[0].pos.z >= near_z && sub[1].pos.z >= near_z &&
                                      sub[2].pos.z >= near_z && sub[3].pos.z >= near_z;
            if( all_in_front ) {
                const uint32_t a = shared( i00 );
                const uint32_t b = shared( i10 );
                const uint32_t c = shared( i11 );
                const uint32_t d = shared( i01 );
                out.indices.insert( out.indices.end(), { a, b, c, a, c, d } );
                triangles += 2;
                continue;
            }
            const bool all_behind = sub[0].pos.z < near_z && sub[1].pos.z < near_z &&
                                    sub[2].pos.z < near_z && sub[3].pos.z < near_z;
            if( all_behind ) {
                continue;
            }
            view_vertex clipped[5];
            const int count = clip_near( sub, 4, near_z, clipped );
            if( count < 3 ) {
                continue;
            }
            const uint32_t first = emit( clipped[0] );
            uint32_t prev = emit( clipped[1] );
            for( int k = 2; k < count; ++k ) {
                const uint32_t cur = emit( clipped[k] );
                out.indices.insert( out.indices.end(), { first, prev, cur } );
                prev = cur;
                ++triangles;
            }
        }
    }
    return triangles;
}

void suggest_subdivisions( const camera &cam, const quad &q, int max_subdivisions, int &nu,
                           int &nv )
{
    max_subdivisions = std::max( max_subdivisions, 1 );
    const float near_z = cam.near_plane();
    vec3 view[4];
    bool crosses_near = false;
    for( int i = 0; i < 4; ++i ) {
        view[i] = cam.to_view( q.corner[i] );
        crosses_near = crosses_near || view[i].z < near_z;
    }
    if( crosses_near ) {
        // Part of the quad is (almost) at the camera, where perspective changes the fastest
        nu = max_subdivisions;
        nv = max_subdivisions;
        return;
    }

    const float ratio_u = std::max( depth_ratio( view[0].z, view[1].z, near_z ),
                                    depth_ratio( view[3].z, view[2].z, near_z ) );
    const float ratio_v = std::max( depth_ratio( view[0].z, view[3].z, near_z ),
                                    depth_ratio( view[1].z, view[2].z, near_z ) );
    nu = subdivisions_for_ratio( ratio_u, max_subdivisions );
    nv = subdivisions_for_ratio( ratio_v, max_subdivisions );

    // Pieces smaller than a few pixels gain nothing
    constexpr float min_piece_px = 12.0f;
    float sx[4];
    float sy[4];
    for( int i = 0; i < 4; ++i ) {
        cam.project( view[i], sx[i], sy[i] );
    }
    const auto edge_length = [&]( int a, int b ) {
        return std::hypot( sx[a] - sx[b], sy[a] - sy[b] );
    };
    const float len_u = std::max( edge_length( 0, 1 ), edge_length( 3, 2 ) );
    const float len_v = std::max( edge_length( 0, 3 ), edge_length( 1, 2 ) );
    nu = std::min( nu, std::max( 1, static_cast<int>( std::ceil( len_u / min_piece_px ) ) ) );
    nv = std::min( nv, std::max( 1, static_cast<int>( std::ceil( len_v / min_piece_px ) ) ) );
}

bool faces_camera( const camera &cam, const vec3 &point_on_face, const vec3 &outward_normal )
{
    return dot( outward_normal, cam.position() - point_on_face ) > 0.0f;
}

vec3 facing_right_vector( const camera &cam, const vec3 &anchor )
{
    const float dx = anchor.x - cam.position().x;
    const float dy = anchor.y - cam.position().y;
    const float len = std::hypot( dx, dy );
    if( len < 1e-3f ) {
        return cam.right();
    }
    // Looking along (fx, fy), right is (-fy, fx): north -> east, east -> south
    return { -dy / len, dx / len, 0.0f };
}

quad make_billboard( const camera &cam, const vec3 &anchor, float left, float right,
                     float bottom, float top, bool flip_u )
{
    const vec3 r = facing_right_vector( cam, anchor );
    const vec3 up = { 0.0f, 0.0f, 1.0f };
    quad q;
    q.corner[0] = anchor + r * left + up * top;
    q.corner[1] = anchor + r * right + up * top;
    q.corner[2] = anchor + r * right + up * bottom;
    q.corner[3] = anchor + r * left + up * bottom;
    if( flip_u ) {
        q.u[0] = 1.0f;
        q.u[1] = 0.0f;
        q.u[2] = 0.0f;
        q.u[3] = 1.0f;
    }
    return q;
}

vec3 face_normal( face f )
{
    switch( f ) {
        case face::north:
            return { 0.0f, -1.0f, 0.0f };
        case face::south:
            return { 0.0f, 1.0f, 0.0f };
        case face::east:
            return { 1.0f, 0.0f, 0.0f };
        case face::west:
            return { -1.0f, 0.0f, 0.0f };
        case face::top:
            return { 0.0f, 0.0f, 1.0f };
        case face::bottom:
            return { 0.0f, 0.0f, -1.0f };
    }
    return { 0.0f, 0.0f, 1.0f };
}

quad make_box_face( const vec3 &lo, const vec3 &hi, face f )
{
    quad q;
    switch( f ) {
        case face::north:
            q.corner[0] = { hi.x, lo.y, hi.z };
            q.corner[1] = { lo.x, lo.y, hi.z };
            q.corner[2] = { lo.x, lo.y, lo.z };
            q.corner[3] = { hi.x, lo.y, lo.z };
            break;
        case face::south:
            q.corner[0] = { lo.x, hi.y, hi.z };
            q.corner[1] = { hi.x, hi.y, hi.z };
            q.corner[2] = { hi.x, hi.y, lo.z };
            q.corner[3] = { lo.x, hi.y, lo.z };
            break;
        case face::east:
            q.corner[0] = { hi.x, hi.y, hi.z };
            q.corner[1] = { hi.x, lo.y, hi.z };
            q.corner[2] = { hi.x, lo.y, lo.z };
            q.corner[3] = { hi.x, hi.y, lo.z };
            break;
        case face::west:
            q.corner[0] = { lo.x, lo.y, hi.z };
            q.corner[1] = { lo.x, hi.y, hi.z };
            q.corner[2] = { lo.x, hi.y, lo.z };
            q.corner[3] = { lo.x, lo.y, lo.z };
            break;
        case face::top:
            q.corner[0] = { lo.x, lo.y, hi.z };
            q.corner[1] = { hi.x, lo.y, hi.z };
            q.corner[2] = { hi.x, hi.y, hi.z };
            q.corner[3] = { lo.x, hi.y, hi.z };
            break;
        case face::bottom:
            // Seen from below, with north at the top of the view, west is on the right
            q.corner[0] = { hi.x, lo.y, lo.z };
            q.corner[1] = { lo.x, lo.y, lo.z };
            q.corner[2] = { lo.x, hi.y, lo.z };
            q.corner[3] = { hi.x, hi.y, lo.z };
            break;
    }
    return q;
}

bool may_be_visible( const camera &cam, const vec3 &world, float radius )
{
    const vec3 v = cam.to_view( world );
    if( v.z < -radius ) {
        return false;
    }
    const float ht = cam.half_tan_fov();
    return std::fabs( v.x ) - radius * std::sqrt( 1.0f + ht * ht ) <= v.z * ht;
}

const std::vector<cell_offset> &far_to_near_offsets( int radius )
{
    static std::vector<cell_offset> cache;
    static int cached_radius = -1;
    radius = std::max( radius, 0 );
    if( cached_radius == radius ) {
        return cache;
    }
    cache.clear();
    for( int dy = -radius; dy <= radius; ++dy ) {
        for( int dx = -radius; dx <= radius; ++dx ) {
            if( dx * dx + dy * dy <= radius * radius ) {
                cache.push_back( { dx, dy } );
            }
        }
    }
    // A cell can only be hidden by cells that are closer to the camera cell along both axes,
    // and those have a strictly smaller manhattan distance.
    std::stable_sort( cache.begin(), cache.end(), []( const cell_offset & a, const cell_offset & b ) {
        return std::abs( a.dx ) + std::abs( a.dy ) > std::abs( b.dx ) + std::abs( b.dy );
    } );
    cached_radius = radius;
    return cache;
}

float distance_shade( float depth, float fade_start, float fade_end, float min_brightness )
{
    if( fade_end <= fade_start ) {
        return 1.0f;
    }
    const float t = std::clamp( ( depth - fade_start ) / ( fade_end - fade_start ), 0.0f, 1.0f );
    return 1.0f - ( 1.0f - min_brightness ) * t;
}

} // namespace render3d
