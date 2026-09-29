#pragma once
#ifndef CATA_SRC_RENDER3D_H
#define CATA_SRC_RENDER3D_H

#include <cstddef>
#include <cstdint>
#include <vector>

/**
 * A small perspective projection toolkit used by the 3D tile view.
 *
 * It knows nothing about SDL, tilesets or the map: it turns textured quads that live in world
 * space into screen space triangles, and it knows in which order cells of a grid have to be drawn
 * so that nearer cells paint over farther ones (there is no depth buffer, the renderer is a
 * painter's algorithm one).
 *
 * World space uses map axes: +x is east, +y is south and +z is up. One unit is one tile.
 * View space has +x to the right, +y up and +z away from the camera (the depth).
 * Yaw is measured clockwise from north, in degrees, pitch is positive when looking up.
 */
namespace render3d
{

struct vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

inline vec3 operator+( const vec3 &a, const vec3 &b )
{
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}
inline vec3 operator-( const vec3 &a, const vec3 &b )
{
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}
inline vec3 operator*( const vec3 &a, float s )
{
    return { a.x * s, a.y * s, a.z * s };
}
inline float dot( const vec3 &a, const vec3 &b )
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline vec3 lerp( const vec3 &a, const vec3 &b, float t )
{
    return a + ( b - a ) * t;
}

/** A vertex that has been moved to view space and carries its texture coordinates. */
struct view_vertex {
    vec3 pos;
    float u = 0.0f;
    float v = 0.0f;
};

/** A projected vertex, in viewport pixels. Depth is the view space distance along the view axis. */
struct screen_vertex {
    float x = 0.0f;
    float y = 0.0f;
    float u = 0.0f;
    float v = 0.0f;
    float depth = 0.0f;
};

/** A pinhole camera. */
class camera
{
    public:
        camera() : camera( vec3(), 0.0f, 0.0f, 90.0f, 640, 480 ) {}
        /**
         * @param position eye position in world space
         * @param yaw_deg heading in degrees, clockwise from north (0 = north, 90 = east)
         * @param pitch_deg elevation in degrees, positive looks up
         * @param fov_deg horizontal field of view in degrees
         * @param viewport_w viewport width in pixels
         * @param viewport_h viewport height in pixels
         */
        camera( const vec3 &position, float yaw_deg, float pitch_deg, float fov_deg,
                int viewport_w, int viewport_h );

        const vec3 &position() const {
            return position_;
        }
        const vec3 &forward() const {
            return forward_;
        }
        const vec3 &right() const {
            return right_;
        }
        const vec3 &up() const {
            return up_;
        }
        int viewport_width() const {
            return viewport_w_;
        }
        int viewport_height() const {
            return viewport_h_;
        }
        float focal_length() const {
            return focal_;
        }
        float near_plane() const {
            return near_;
        }
        /** Half of the tangent of the horizontal field of view. */
        float half_tan_fov() const {
            return half_tan_fov_;
        }

        /** World to view space. */
        vec3 to_view( const vec3 &world ) const;
        /** Projects a view space point. The point must be in front of the camera (z > 0). */
        void project( const vec3 &view, float &sx, float &sy ) const;
        /**
         * Casts a ray through a viewport pixel. Returns its (normalized) direction in world space.
         */
        vec3 pixel_direction( float sx, float sy ) const;

    private:
        vec3 position_;
        vec3 forward_;
        vec3 right_;
        vec3 up_;
        int viewport_w_ = 0;
        int viewport_h_ = 0;
        float focal_ = 1.0f;
        float half_tan_fov_ = 1.0f;
        float near_ = 0.05f;
};

/**
 * A textured quad in world space. Corners go around the perimeter, and are named as the
 * corners of the sprite they carry: top left, top right, bottom right, bottom left.
 */
struct quad {
    vec3 corner[4];
    float u[4] = { 0.0f, 1.0f, 1.0f, 0.0f };
    float v[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
};

/**
 * Grows a quad on every side by (about) the given number of pixels, as seen at its middle. The
 * texture coordinates stay as they are, the sprite is stretched a little. Quads that are meant to
 * meet edge to edge overlap this way, no hairline gaps appear between them where a rasterizer
 * rounds their edges differently.
 */
void grow_quad( const camera &cam, quad &q, float pixels );

/** Triangles ready for the renderer. */
struct mesh {
    std::vector<screen_vertex> vertices;
    std::vector<uint32_t> indices;
    void clear() {
        vertices.clear();
        indices.clear();
    }
};

/**
 * Clips a convex polygon against the camera's near plane (Sutherland-Hodgman) and keeps the part
 * with view depth >= near_z. Texture coordinates are interpolated along the cut edges.
 * @param out must have room for in_count + 1 vertices
 * @return the number of vertices written to out (0 when everything was clipped away)
 */
int clip_near( const view_vertex *in, int in_count, float near_z, view_vertex *out );

/**
 * Projects a quad, subdividing it into nu by nv sub quads, and appends the resulting triangles
 * to the mesh. The screen is affine in texture space, so subdividing is how the perspective
 * correct look is approximated. Every sub quad is clipped against the near plane on its own.
 * @return the number of triangles appended
 */
int tessellate( const camera &cam, const quad &q, int nu, int nv, mesh &out );

/**
 * Chooses how many times each axis of the quad should be split so that affine texture
 * interpolation stays close to the perspective correct result.
 * @param max_subdivisions upper bound for each of nu and nv (at least 1)
 */
void suggest_subdivisions( const camera &cam, const quad &q, int max_subdivisions, int &nu,
                           int &nv );

/** True when a face with the given outward normal, through the given point, is seen from the front. */
bool faces_camera( const camera &cam, const vec3 &point_on_face, const vec3 &outward_normal );

/**
 * Horizontal unit vector, perpendicular to the direction from the camera to the anchor, that
 * points to the right as seen from the camera. Falls back to the camera's own right vector when
 * the anchor is (nearly) straight above or below the camera.
 */
vec3 facing_right_vector( const camera &cam, const vec3 &anchor );

/**
 * A vertical quad that stands at the anchor and faces the camera (a billboard).
 * @param anchor the point on the ground the sprite stands on
 * @param left signed offset of the left edge from the anchor, along the quad's right axis
 * @param right signed offset of the right edge
 * @param bottom height of the bottom edge above the anchor
 * @param top height of the top edge above the anchor
 * @param flip_u mirror the sprite horizontally
 */
quad make_billboard( const camera &cam, const vec3 &anchor, float left, float right, float bottom,
                     float top, bool flip_u = false );

/** The faces of an axis aligned box. */
enum class face {
    north, // -y
    south, // +y
    east, // +x
    west, // -x
    top, // +z
    bottom // -z
};

/** Outward pointing unit normal of a face. */
vec3 face_normal( face f );

/**
 * A face of the axis aligned box spanning lo to hi. Seen from the outside the sprite is upright
 * and not mirrored: on the side faces the sprite's up is up, on the top and bottom faces it is
 * north.
 */
quad make_box_face( const vec3 &lo, const vec3 &hi, face f );

/**
 * True when a sphere of the given radius around the world position may be inside the horizontal
 * field of view (and not behind the camera). Conservative: it never culls what could be seen.
 */
bool may_be_visible( const camera &cam, const vec3 &world, float radius );

/** A cell offset relative to the camera cell. */
struct cell_offset {
    int dx;
    int dy;
};

/**
 * All cell offsets within radius (inclusive, euclidean), sorted so that drawing them in order
 * paints farther cells first. Whenever a cell can hide another one, from any camera position
 * inside the camera cell, it comes after it in this list. The list is cached for the last radius
 * asked for.
 */
const std::vector<cell_offset> &far_to_near_offsets( int radius );

/** Brightness multiplier in [min_brightness, 1] for a point at the given depth. */
float distance_shade( float depth, float fade_start, float fade_end, float min_brightness );

} // namespace render3d

#endif // CATA_SRC_RENDER3D_H
