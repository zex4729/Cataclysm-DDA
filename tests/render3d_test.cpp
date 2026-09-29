#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "cata_catch.h"
#include "render3d.h"

using namespace render3d;

namespace
{

constexpr float pi = 3.14159265358979323846f;

camera make_camera( const vec3 &pos, float yaw = 0.0f, float pitch = 0.0f )
{
    return camera( pos, yaw, pitch, 90.0f, 640, 480 );
}

// Heading (clockwise from north, in degrees) that makes the camera look along dir
float yaw_towards( const vec3 &dir )
{
    return std::atan2( dir.x, -dir.y ) * 180.0f / pi;
}

void project_point( const camera &cam, const vec3 &world, float &sx, float &sy )
{
    const vec3 v = cam.to_view( world );
    REQUIRE( v.z > 0.0f );
    cam.project( v, sx, sy );
}

} // namespace

TEST_CASE( "render3d_camera_basis", "[render3d]" )
{
    SECTION( "north" ) {
        const camera cam = make_camera( vec3(), 0.0f );
        CHECK( cam.forward().x == Approx( 0.0f ).margin( 1e-5 ) );
        CHECK( cam.forward().y == Approx( -1.0f ) );
        CHECK( cam.right().x == Approx( 1.0f ) );
        CHECK( cam.right().y == Approx( 0.0f ).margin( 1e-5 ) );
        CHECK( cam.up().z == Approx( 1.0f ) );
    }
    SECTION( "east" ) {
        const camera cam = make_camera( vec3(), 90.0f );
        CHECK( cam.forward().x == Approx( 1.0f ) );
        CHECK( cam.forward().y == Approx( 0.0f ).margin( 1e-5 ) );
        // Facing east, south is on the right hand
        CHECK( cam.right().y == Approx( 1.0f ) );
    }
    SECTION( "south" ) {
        const camera cam = make_camera( vec3(), 180.0f );
        CHECK( cam.forward().y == Approx( 1.0f ) );
        CHECK( cam.right().x == Approx( -1.0f ) );
    }
    SECTION( "looking up tilts the up vector back" ) {
        const camera cam = make_camera( vec3(), 0.0f, 45.0f );
        CHECK( cam.forward().z == Approx( std::sqrt( 0.5f ) ) );
        CHECK( cam.up().z == Approx( std::sqrt( 0.5f ) ) );
        CHECK( dot( cam.forward(), cam.up() ) == Approx( 0.0f ).margin( 1e-5 ) );
        CHECK( dot( cam.forward(), cam.right() ) == Approx( 0.0f ).margin( 1e-5 ) );
    }
}

TEST_CASE( "render3d_projection", "[render3d]" )
{
    const camera cam = make_camera( vec3{ 10.0f, 10.0f, 0.5f }, 0.0f );
    float sx = 0.0f;
    float sy = 0.0f;

    SECTION( "straight ahead lands in the middle of the viewport" ) {
        project_point( cam, vec3{ 10.0f, 5.0f, 0.5f }, sx, sy );
        CHECK( sx == Approx( 320.0f ) );
        CHECK( sy == Approx( 240.0f ) );
    }
    SECTION( "east is on the right, up is up" ) {
        project_point( cam, vec3{ 12.0f, 5.0f, 0.5f }, sx, sy );
        CHECK( sx > 320.0f );
        project_point( cam, vec3{ 8.0f, 5.0f, 0.5f }, sx, sy );
        CHECK( sx < 320.0f );
        project_point( cam, vec3{ 10.0f, 5.0f, 1.5f }, sx, sy );
        CHECK( sy < 240.0f );
    }
    SECTION( "a 90 degree field of view puts the edge of the view at half the width" ) {
        // 5 tiles ahead and 5 tiles to the side is exactly 45 degrees off axis
        project_point( cam, vec3{ 15.0f, 5.0f, 0.5f }, sx, sy );
        CHECK( sx == Approx( 640.0f ) );
    }
    SECTION( "farther things are smaller" ) {
        float near_x = 0.0f;
        float far_x = 0.0f;
        project_point( cam, vec3{ 11.0f, 8.0f, 0.5f }, near_x, sy );
        project_point( cam, vec3{ 11.0f, 0.0f, 0.5f }, far_x, sy );
        CHECK( near_x - 320.0f > far_x - 320.0f );
    }
    SECTION( "pixel_direction inverts the projection" ) {
        const vec3 target = { 12.0f, 4.0f, 1.25f };
        project_point( cam, target, sx, sy );
        const vec3 dir = cam.pixel_direction( sx, sy );
        const vec3 to_target = target - cam.position();
        const float len = std::sqrt( dot( to_target, to_target ) );
        CHECK( dir.x == Approx( to_target.x / len ) );
        CHECK( dir.y == Approx( to_target.y / len ) );
        CHECK( dir.z == Approx( to_target.z / len ) );
    }
}

TEST_CASE( "render3d_clip_near", "[render3d]" )
{
    const float near_z = 1.0f;
    view_vertex out[5];

    SECTION( "fully in front is untouched" ) {
        const view_vertex tri[3] = { { { 0, 0, 2 }, 0, 0 }, { { 1, 0, 2 }, 1, 0 }, { { 0, 1, 3 }, 0, 1 } };
        CHECK( clip_near( tri, 3, near_z, out ) == 3 );
    }
    SECTION( "fully behind is removed" ) {
        const view_vertex tri[3] = { { { 0, 0, 0 }, 0, 0 }, { { 1, 0, 0.5f }, 1, 0 }, { { 0, 1, -1 }, 0, 1 } };
        CHECK( clip_near( tri, 3, near_z, out ) == 0 );
    }
    SECTION( "one corner behind turns a triangle into a quad" ) {
        const view_vertex tri[3] = { { { 0, 0, 3 }, 0, 0 }, { { 1, 0, 3 }, 1, 0 }, { { 0, 1, -1 }, 0, 1 } };
        const int n = clip_near( tri, 3, near_z, out );
        REQUIRE( n == 4 );
        for( int i = 0; i < n; ++i ) {
            CHECK( out[i].pos.z >= near_z - 1e-6f );
        }
        // The cut edge from (0,1,-1) to (0,0,3) crosses z = 1 halfway: v = 0.5
        bool found = false;
        for( int i = 0; i < n; ++i ) {
            if( out[i].pos.z == Approx( near_z ) && out[i].pos.y == Approx( 0.5f ) ) {
                found = true;
                CHECK( out[i].v == Approx( 0.5f ) );
            }
        }
        CHECK( found );
    }
    SECTION( "a quad with two corners behind stays a quad" ) {
        const view_vertex quad_in[4] = {
            { { 0, 0, 4 }, 0, 0 }, { { 1, 0, 4 }, 1, 0 }, { { 1, 0, -2 }, 1, 1 }, { { 0, 0, -2 }, 0, 1 }
        };
        const int n = clip_near( quad_in, 4, near_z, out );
        CHECK( n == 4 );
        for( int i = 0; i < n; ++i ) {
            CHECK( out[i].pos.z >= near_z - 1e-6f );
        }
    }
}

TEST_CASE( "render3d_tessellate", "[render3d]" )
{
    // A 1x1 wall face 4 tiles north of the camera, spanning the camera's height
    const camera cam = make_camera( vec3{ 0.5f, 0.5f, 0.5f }, 0.0f );
    const quad wall = make_box_face( vec3{ 0.0f, -4.0f, 0.0f }, vec3{ 1.0f, -3.0f, 1.0f },
                                     face::south );

    SECTION( "an unsplit quad is two triangles" ) {
        mesh m;
        CHECK( tessellate( cam, wall, 1, 1, m ) == 2 );
        CHECK( m.vertices.size() == 4 );
        CHECK( m.indices.size() == 6 );
    }
    SECTION( "sub quads share their corners" ) {
        mesh m;
        CHECK( tessellate( cam, wall, 3, 2, m ) == 12 );
        CHECK( m.vertices.size() == 4 * 3 );
        CHECK( m.indices.size() == 36 );
        for( const uint32_t i : m.indices ) {
            CHECK( i < m.vertices.size() );
        }
    }
    SECTION( "texture coordinates cover the whole sprite" ) {
        mesh m;
        tessellate( cam, wall, 2, 2, m );
        float min_u = 1.0f;
        float max_u = 0.0f;
        float min_v = 1.0f;
        float max_v = 0.0f;
        for( const screen_vertex &v : m.vertices ) {
            min_u = std::min( min_u, v.u );
            max_u = std::max( max_u, v.u );
            min_v = std::min( min_v, v.v );
            max_v = std::max( max_v, v.v );
        }
        CHECK( min_u == Approx( 0.0f ) );
        CHECK( max_u == Approx( 1.0f ) );
        CHECK( min_v == Approx( 0.0f ) );
        CHECK( max_v == Approx( 1.0f ) );
    }
    SECTION( "top left of the sprite is at the top left of the screen" ) {
        mesh m;
        tessellate( cam, wall, 1, 1, m );
        REQUIRE( m.vertices.size() == 4 );
        const screen_vertex &tl = m.vertices[0];
        const screen_vertex &tr = m.vertices[1];
        const screen_vertex &bl = m.vertices[3];
        CHECK( tl.u == Approx( 0.0f ) );
        CHECK( tl.v == Approx( 0.0f ) );
        CHECK( tl.x < tr.x );
        CHECK( tl.y < bl.y );
    }
    SECTION( "nothing is left when the quad is behind the camera" ) {
        const camera other = make_camera( vec3{ 0.5f, 0.5f, 0.5f }, 180.0f );
        mesh m;
        CHECK( tessellate( other, wall, 4, 4, m ) == 0 );
        CHECK( m.vertices.empty() );
        CHECK( m.indices.empty() );
    }
    SECTION( "a quad that reaches behind the camera is cut at the near plane" ) {
        // A floor tile that stretches from far ahead to well behind the camera
        quad floor;
        floor.corner[0] = { -1.0f, -20.0f, 0.0f };
        floor.corner[1] = { 2.0f, -20.0f, 0.0f };
        floor.corner[2] = { 2.0f, 6.0f, 0.0f };
        floor.corner[3] = { -1.0f, 6.0f, 0.0f };
        mesh m;
        const int triangles = tessellate( cam, floor, 2, 4, m );
        CHECK( triangles > 0 );
        for( const screen_vertex &v : m.vertices ) {
            CHECK( v.depth >= cam.near_plane() - 1e-5f );
            CHECK( std::isfinite( v.x ) );
            CHECK( std::isfinite( v.y ) );
        }
        for( const uint32_t i : m.indices ) {
            CHECK( i < m.vertices.size() );
        }
    }
}

TEST_CASE( "render3d_grow_quad_overlaps_neighbours", "[render3d]" )
{
    const camera cam = make_camera( vec3{ 0.5f, 0.5f, 0.5f }, 0.0f );
    const quad floor = make_box_face( vec3{ 0.0f, -6.0f, 0.0f }, vec3{ 1.0f, -5.0f, 0.0f }, face::top );

    // Measured across the middle of the quad
    const auto width_on_screen = [&]( const quad & q ) {
        float left = 0.0f;
        float right = 0.0f;
        float y = 0.0f;
        project_point( cam, lerp( q.corner[0], q.corner[3], 0.5f ), left, y );
        project_point( cam, lerp( q.corner[1], q.corner[2], 0.5f ), right, y );
        return right - left;
    };
    const auto height_on_screen = [&]( const quad & q ) {
        float x = 0.0f;
        float top = 0.0f;
        float bottom = 0.0f;
        project_point( cam, q.corner[0], x, top );
        project_point( cam, q.corner[3], x, bottom );
        return bottom - top;
    };

    quad grown = floor;
    grow_quad( cam, grown, 1.0f );
    // A pixel more on each side, seen at the middle of the quad
    CHECK( width_on_screen( grown ) == Approx( width_on_screen( floor ) + 2.0f ).margin( 0.2 ) );
    CHECK( height_on_screen( grown ) == Approx( height_on_screen( floor ) + 2.0f ).margin( 0.4 ) );
    for( int i = 0; i < 4; ++i ) {
        // The sprite is stretched, not shifted
        CHECK( grown.u[i] == floor.u[i] );
        CHECK( grown.v[i] == floor.v[i] );
    }
    // Neighbouring tiles now overlap
    const quad next = make_box_face( vec3{ 1.0f, -6.0f, 0.0f }, vec3{ 2.0f, -5.0f, 0.0f }, face::top );
    quad grown_next = next;
    grow_quad( cam, grown_next, 1.0f );
    CHECK( grown.corner[1].x > grown_next.corner[0].x );

    SECTION( "a quad at the camera is left alone" ) {
        quad at_camera = make_box_face( vec3{ 0.0f, 0.0f, 0.5f }, vec3{ 1.0f, 1.0f, 0.5f }, face::top );
        const quad before = at_camera;
        grow_quad( make_camera( vec3{ 0.5f, 0.5f, 0.5f }, 180.0f ), at_camera, 1.0f );
        for( int i = 0; i < 4; ++i ) {
            CHECK( at_camera.corner[i].x == before.corner[i].x );
            CHECK( at_camera.corner[i].y == before.corner[i].y );
        }
    }
    SECTION( "the growth is limited" ) {
        // Far away a pixel is a large part of a tile
        const quad far_tile = make_box_face( vec3{ 0.0f, -400.0f, 0.0f }, vec3{ 1.0f, -399.0f, 0.0f },
                                             face::top );
        quad grown_far = far_tile;
        grow_quad( cam, grown_far, 1.0f );
        CHECK( grown_far.corner[1].x - grown_far.corner[0].x <= 1.5f + 1e-4f );
    }
}

TEST_CASE( "render3d_subdivisions", "[render3d]" )
{
    const camera cam = make_camera( vec3{ 0.5f, 0.5f, 0.5f }, 0.0f );

    SECTION( "far away quads are not split" ) {
        const quad far_wall = make_box_face( vec3{ 0.0f, -40.0f, 0.0f }, vec3{ 1.0f, -39.0f, 1.0f },
                                             face::south );
        int nu = 0;
        int nv = 0;
        suggest_subdivisions( cam, far_wall, 8, nu, nv );
        CHECK( nu == 1 );
        CHECK( nv == 1 );
    }
    SECTION( "a floor tile reaching toward the camera is split along its depth" ) {
        const quad floor = make_box_face( vec3{ 0.0f, -2.0f, 0.0f }, vec3{ 1.0f, -1.0f, 0.0f },
                                          face::top );
        int nu = 0;
        int nv = 0;
        suggest_subdivisions( cam, floor, 8, nu, nv );
        CHECK( nu == 1 );
        CHECK( nv > 1 );
        CHECK( nv <= 8 );
    }
    SECTION( "the cap is honored" ) {
        const quad floor = make_box_face( vec3{ 0.0f, -1.0f, 0.0f }, vec3{ 1.0f, 0.0f, 0.0f },
                                          face::top );
        int nu = 0;
        int nv = 0;
        suggest_subdivisions( cam, floor, 3, nu, nv );
        CHECK( nu <= 3 );
        CHECK( nv <= 3 );
        CHECK( nu >= 1 );
        CHECK( nv >= 1 );
    }
}

TEST_CASE( "render3d_box_faces_are_upright_and_unmirrored_from_outside", "[render3d]" )
{
    const vec3 lo = { 10.0f, 20.0f, 0.0f };
    const vec3 hi = { 11.0f, 21.0f, 1.0f };
    const vec3 center = { 10.5f, 20.5f, 0.5f };

    const face faces[] = { face::north, face::south, face::east, face::west, face::top };
    for( const face f : faces ) {
        const vec3 normal = face_normal( f );
        CAPTURE( static_cast<int>( f ) );
        const quad q = make_box_face( lo, hi, f );

        vec3 cam_pos = center + normal * 3.0f;
        float yaw = 0.0f;
        float pitch = 0.0f;
        if( f == face::top ) {
            // Looking straight down with north at the top of the screen
            yaw = 0.0f;
            pitch = -89.0f;
        } else {
            yaw = yaw_towards( normal * -1.0f );
        }
        const camera cam = make_camera( cam_pos, yaw, pitch );
        REQUIRE( faces_camera( cam, q.corner[0], normal ) );

        float sx[4];
        float sy[4];
        for( int i = 0; i < 4; ++i ) {
            project_point( cam, q.corner[i], sx[i], sy[i] );
        }
        CHECK( sx[0] < sx[1] );
        CHECK( sx[3] < sx[2] );
        CHECK( sy[0] < sy[3] );
        CHECK( sy[1] < sy[2] );
    }

    SECTION( "the face on the far side of the box is not facing the camera" ) {
        const camera cam = make_camera( center + vec3{ 0.0f, -3.0f, 0.0f }, 180.0f );
        CHECK( faces_camera( cam, lo, face_normal( face::north ) ) );
        CHECK_FALSE( faces_camera( cam, hi, face_normal( face::south ) ) );
    }
    SECTION( "the bottom face is upright seen from below" ) {
        const vec3 cam_pos = center - vec3{ 0.0f, 0.0f, 3.0f };
        const camera cam = make_camera( cam_pos, 0.0f, 89.0f );
        const quad q = make_box_face( lo, hi, face::bottom );
        REQUIRE( faces_camera( cam, q.corner[0], face_normal( face::bottom ) ) );
        float sx[4];
        float sy[4];
        for( int i = 0; i < 4; ++i ) {
            project_point( cam, q.corner[i], sx[i], sy[i] );
        }
        // Looking up with the head toward north, the top of the screen is south. Whatever the
        // camera's roll, the sprite must not be mirrored: its winding on screen matches the
        // winding of the other faces (top left -> top right -> bottom right is clockwise).
        const float winding = ( sx[1] - sx[0] ) * ( sy[2] - sy[1] ) -
                              ( sy[1] - sy[0] ) * ( sx[2] - sx[1] );
        CHECK( winding > 0.0f );
    }
}

TEST_CASE( "render3d_billboards_face_the_camera", "[render3d]" )
{
    const camera cam = make_camera( vec3{ 5.5f, 5.5f, 0.5f }, 0.0f );

    SECTION( "right vector" ) {
        // The anchor is south of the camera: looking south, right is west
        const vec3 r = facing_right_vector( cam, vec3{ 5.5f, 9.5f, 0.0f } );
        CHECK( r.x == Approx( -1.0f ) );
        CHECK( r.y == Approx( 0.0f ).margin( 1e-5 ) );
        CHECK( r.z == Approx( 0.0f ) );
        // North of the camera: looking north, right is east
        const vec3 r2 = facing_right_vector( cam, vec3{ 5.5f, 1.5f, 0.0f } );
        CHECK( r2.x == Approx( 1.0f ) );
    }
    SECTION( "right vector falls back when the anchor is at the camera" ) {
        const vec3 r = facing_right_vector( cam, vec3{ 5.5f, 5.5f, 0.0f } );
        CHECK( r.x == Approx( cam.right().x ) );
        CHECK( r.y == Approx( cam.right().y ).margin( 1e-5 ) );
    }
    SECTION( "a billboard is perpendicular to the line of sight and vertical" ) {
        const vec3 anchor = { 8.5f, 2.5f, 0.0f };
        const quad q = make_billboard( cam, anchor, -0.5f, 0.5f, 0.0f, 1.0f );
        // Vertical: the top and bottom corners share x and y
        CHECK( q.corner[0].x == Approx( q.corner[3].x ) );
        CHECK( q.corner[0].y == Approx( q.corner[3].y ) );
        CHECK( q.corner[0].z == Approx( 1.0f ) );
        CHECK( q.corner[3].z == Approx( 0.0f ) );
        // Its width axis is perpendicular to the horizontal direction to the camera
        const vec3 across = q.corner[1] - q.corner[0];
        const vec3 to_cam = { cam.position().x - anchor.x, cam.position().y - anchor.y, 0.0f };
        CHECK( dot( across, to_cam ) == Approx( 0.0f ).margin( 1e-4 ) );
        CHECK( std::sqrt( dot( across, across ) ) == Approx( 1.0f ) );
        // Centered on the anchor
        CHECK( ( q.corner[0].x + q.corner[1].x ) * 0.5f == Approx( anchor.x ) );
        CHECK( ( q.corner[0].y + q.corner[1].y ) * 0.5f == Approx( anchor.y ) );
    }
    SECTION( "left is left and right is right on screen, whichever way the camera looks" ) {
        const float yaws[] = { 0.0f, 37.0f, 90.0f, 180.0f, 251.0f };
        for( const float yaw : yaws ) {
            CAPTURE( yaw );
            const float rad = yaw * pi / 180.0f;
            const vec3 forward = { std::sin( rad ), -std::cos( rad ), 0.0f };
            const vec3 pos = { 5.5f, 5.5f, 0.5f };
            const camera c = make_camera( pos, yaw );
            const quad q = make_billboard( c, pos + forward * 6.0f, -0.5f, 0.5f, 0.0f, 1.0f );
            float sx[4];
            float sy[4];
            for( int i = 0; i < 4; ++i ) {
                project_point( c, q.corner[i], sx[i], sy[i] );
            }
            CHECK( sx[0] < sx[1] );
            CHECK( sy[0] < sy[3] );
            CHECK( sx[0] == Approx( sx[3] ) );
        }
    }
    SECTION( "flipping mirrors the texture coordinates" ) {
        const quad q = make_billboard( cam, vec3{ 8.5f, 2.5f, 0.0f }, -0.5f, 0.5f, 0.0f, 1.0f, true );
        CHECK( q.u[0] == Approx( 1.0f ) );
        CHECK( q.u[1] == Approx( 0.0f ) );
        CHECK( q.u[2] == Approx( 0.0f ) );
        CHECK( q.u[3] == Approx( 1.0f ) );
    }
}

TEST_CASE( "render3d_frustum_culling_never_hides_visible_cells", "[render3d]" )
{
    const camera cam = make_camera( vec3{ 0.5f, 0.5f, 0.5f }, 30.0f );
    for( int y = -12; y <= 12; ++y ) {
        for( int x = -12; x <= 12; ++x ) {
            const vec3 center = { static_cast<float>( x ) + 0.5f, static_cast<float>( y ) + 0.5f, 0.0f };
            bool any_corner_visible = false;
            for( int c = 0; c < 4; ++c ) {
                const float corner_x = center.x + ( c % 2 ? 0.5f : -0.5f );
                const float corner_y = center.y + ( c / 2 ? 0.5f : -0.5f );
                const vec3 corner = { corner_x, corner_y, 0.0f };
                const vec3 v = cam.to_view( corner );
                if( v.z > cam.near_plane() && std::fabs( v.x ) <= v.z * cam.half_tan_fov() ) {
                    any_corner_visible = true;
                }
            }
            if( any_corner_visible ) {
                CAPTURE( x, y );
                CHECK( may_be_visible( cam, center, 1.5f ) );
            }
        }
    }
    SECTION( "things behind the camera are culled" ) {
        CHECK_FALSE( may_be_visible( cam, cam.position() - cam.forward() * 10.0f, 1.5f ) );
    }
}

TEST_CASE( "render3d_cell_order_paints_far_cells_first", "[render3d]" )
{
    constexpr int radius = 7;
    const std::vector<cell_offset> &order = far_to_near_offsets( radius );

    SECTION( "covers the disc exactly once" ) {
        size_t expected = 0;
        for( int dy = -radius; dy <= radius; ++dy ) {
            for( int dx = -radius; dx <= radius; ++dx ) {
                if( dx * dx + dy * dy <= radius * radius ) {
                    ++expected;
                }
            }
        }
        CHECK( order.size() == expected );
        std::vector<int> seen( ( 2 * radius + 1 ) * ( 2 * radius + 1 ), 0 );
        for( const cell_offset &c : order ) {
            ++seen[( c.dy + radius ) * ( 2 * radius + 1 ) + c.dx + radius];
        }
        CHECK( *std::max_element( seen.begin(), seen.end() ) == 1 );
    }
    SECTION( "the camera cell is painted last" ) {
        REQUIRE_FALSE( order.empty() );
        CHECK( order.back().dx == 0 );
        CHECK( order.back().dy == 0 );
    }
    SECTION( "a cell that can hide another one is painted after it" ) {
        std::vector<int> position( ( 2 * radius + 1 ) * ( 2 * radius + 1 ), -1 );
        for( size_t i = 0; i < order.size(); ++i ) {
            position[( order[i].dy + radius ) * ( 2 * radius + 1 ) + order[i].dx + radius] =
                static_cast<int>( i );
        }
        for( const cell_offset &hidden : order ) {
            for( const cell_offset &hider : order ) {
                if( std::abs( hider.dx ) <= std::abs( hidden.dx ) &&
                    std::abs( hider.dy ) <= std::abs( hidden.dy ) &&
                    ( hider.dx != hidden.dx || hider.dy != hidden.dy ) ) {
                    // Only cells in the same quadrant (or on its borders) can hide each other
                    const bool same_x_side = hider.dx == 0 || hidden.dx == 0 || ( hider.dx > 0 ) == ( hidden.dx > 0 );
                    const bool same_y_side = hider.dy == 0 || hidden.dy == 0 || ( hider.dy > 0 ) == ( hidden.dy > 0 );
                    if( same_x_side && same_y_side ) {
                        CAPTURE( hidden.dx, hidden.dy, hider.dx, hider.dy );
                        REQUIRE( position[( hider.dy + radius ) * ( 2 * radius + 1 ) + hider.dx + radius] >
                                 position[( hidden.dy + radius ) * ( 2 * radius + 1 ) + hidden.dx + radius] );
                    }
                }
            }
        }
    }
    SECTION( "a ray really only meets cells in this order" ) {
        // March rays from a few positions inside the camera cell and check that the cells are
        // met in the order the list paints them back to front
        std::vector<int> position( ( 2 * radius + 1 ) * ( 2 * radius + 1 ), -1 );
        for( size_t i = 0; i < order.size(); ++i ) {
            position[( order[i].dy + radius ) * ( 2 * radius + 1 ) + order[i].dx + radius] =
                static_cast<int>( i );
        }
        const vec3 starts[] = { vec3{ 0.5f, 0.5f, 0.0f }, vec3{ 0.05f, 0.9f, 0.0f }, vec3{ 0.95f, 0.1f, 0.0f } };
        for( const vec3 &start : starts ) {
            for( int a = 0; a < 360; a += 3 ) {
                const float rad = static_cast<float>( a ) * pi / 180.0f;
                const float dx = std::cos( rad );
                const float dy = std::sin( rad );
                int previous_position = 1 << 20;
                for( float t = 0.0f; t < static_cast<float>( radius ); t += 0.02f ) {
                    const int cx = static_cast<int>( std::floor( start.x + dx * t ) );
                    const int cy = static_cast<int>( std::floor( start.y + dy * t ) );
                    if( std::abs( cx ) > radius || std::abs( cy ) > radius ) {
                        break;
                    }
                    const int p = position[( cy + radius ) * ( 2 * radius + 1 ) + cx + radius];
                    if( p < 0 ) {
                        break;
                    }
                    CAPTURE( a, cx, cy );
                    // Walking away from the camera we must go towards later positions... that is,
                    // nearer cells (met first) are painted after the farther ones (met later)
                    REQUIRE( p <= previous_position );
                    previous_position = p;
                }
            }
        }
    }
}

TEST_CASE( "render3d_distance_shade", "[render3d]" )
{
    CHECK( distance_shade( 1.0f, 10.0f, 30.0f, 0.25f ) == Approx( 1.0f ) );
    CHECK( distance_shade( 10.0f, 10.0f, 30.0f, 0.25f ) == Approx( 1.0f ) );
    CHECK( distance_shade( 20.0f, 10.0f, 30.0f, 0.25f ) == Approx( 0.625f ) );
    CHECK( distance_shade( 30.0f, 10.0f, 30.0f, 0.25f ) == Approx( 0.25f ) );
    CHECK( distance_shade( 99.0f, 10.0f, 30.0f, 0.25f ) == Approx( 0.25f ) );
    // A degenerate range never darkens
    CHECK( distance_shade( 99.0f, 30.0f, 30.0f, 0.25f ) == Approx( 1.0f ) );
}
