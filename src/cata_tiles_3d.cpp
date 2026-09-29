#if defined(TILES)
#include "cata_tiles.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "avatar.h"
#include "cached_options.h"
#include "cata_scope_helpers.h"
#include "character.h"
#include "coordinates.h"
#include "creature.h"
#include "creature_tracker.h"
#include "enums.h"
#include "game.h"
#include "level_cache.h"
#include "lightmap.h"
#include "map.h"
#include "map_scale_constants.h"
#include "mapdata.h"
#include "mtype.h"
#include "options.h"
#include "point.h"
#include "render3d.h"
#include "sdl_wrappers.h"
#include "sdltiles.h"

/*
 * The 3D view.
 *
 * It reuses everything the flat renderer knows about which sprite goes where and how it is lit.
 * The only thing that changes is what happens to a sprite once it has been picked: draw_sprite_at
 * hands it to draw_sprite_3d, which puts it in the scene as
 *   - a quad lying on the floor for terrain (and traps and fields),
 *   - a block that shows the same sprite on all its sides for terrain that cannot be walked
 *     through,
 *   - a vertical quad that always faces the camera for everything else (creatures, furniture,
 *     vehicle parts and the top item of a pile).
 *
 * There is no depth buffer. Cells are drawn from the farthest to the nearest (see
 * render3d::far_to_near_offsets) and everything in a cell is drawn in the same order as in the flat
 * renderer, so what is nearer paints over what is farther.
 */

namespace
{

// The view is measured in tiles: a tile is one unit wide and a sprite as tall as it is wide is one
// unit high.
constexpr float pi = 3.14159265358979f;
constexpr float wall_height = 1.0f;
constexpr float eye_height = 0.6f;
// Items are drawn smaller than creatures, they are things lying on the floor
constexpr float item_scale = 0.5f;
// Billboards this close to the camera would fill the screen
constexpr float min_billboard_distance = 0.4f;
// Textures are affine on screen, subdividing large quads makes them look right
constexpr int max_subdivisions = 10;
// The sprites live in an atlas: stay clear of the neighbouring sprites when sampling
constexpr float texel_inset = 0.25f;
// Blocks and floor tiles are drawn a little larger than they are, or the rasterizer leaves
// hairline gaps between them
constexpr float seam_overlap_pixels = 0.6f;
// Beyond this fraction of the draw distance the view fades to dark
constexpr float fog_start_fraction = 0.4f;
constexpr float fog_min_brightness = 0.3f;
// A block shows the same sprite on every side, so the sides are shaded to tell them apart: the light
// comes from the south east
constexpr float shade_north = 0.72f;
constexpr float shade_south = 0.9f;
constexpr float shade_east = 1.0f;
constexpr float shade_west = 0.82f;
constexpr float shade_top = 1.0f;
constexpr float shade_bottom = 0.6f;

// Which way the camera looks, in degrees clockwise from north. It is shared by all tile contexts
// and follows the avatar: whenever it steps, the camera turns to face that way.
struct heading_state {
    float degrees = 0.0f;
    std::optional<tripoint_abs_ms> last_avatar_pos;
    // The camera of the last frame, to find out what is under the mouse pointer
    std::optional<render3d::camera> last_camera;
    int last_z = 0;
    int last_radius = 0;
};

heading_state &heading()
{
    static heading_state state;
    return state;
}

void follow_avatar( const tripoint_abs_ms &avatar_pos )
{
    heading_state &h = heading();
    if( h.last_avatar_pos && *h.last_avatar_pos != avatar_pos ) {
        const tripoint_rel_ms step = avatar_pos - *h.last_avatar_pos;
        // Ignore teleports and fast vehicles, the camera only turns for steps
        if( ( step.x() != 0 || step.y() != 0 ) && std::abs( step.x() ) <= 1 &&
            std::abs( step.y() ) <= 1 ) {
            h.degrees = std::atan2( static_cast<float>( step.x() ),
                                    -static_cast<float>( step.y() ) ) * 180.0f / pi;
        }
    }
    h.last_avatar_pos = avatar_pos;
    h.degrees = std::fmod( h.degrees + 360.0f, 360.0f );
}

float face_shade( render3d::face f )
{
    switch( f ) {
        case render3d::face::north:
            return shade_north;
        case render3d::face::south:
            return shade_south;
        case render3d::face::east:
            return shade_east;
        case render3d::face::west:
            return shade_west;
        case render3d::face::top:
            return shade_top;
        case render3d::face::bottom:
            return shade_bottom;
    }
    return 1.0f;
}

enum class placement {
    floor,
    block,
    billboard,
};

// A vertex as SDL wants it: one interleaved array
struct gpu_vertex {
    float x;
    float y;
    SDL_FColor color;
    float u;
    float v;
};

} // namespace

// Everything the 3D view keeps while it draws one frame
struct scene_3d {
    render3d::camera cam;
    // Where the viewport is in the window
    point origin;
    // Distance that is drawn, in tiles
    int radius = 0;

    // What the sprite that is drawn right now is: set by tile_3d_scope
    TILE_CATEGORY category = TILE_CATEGORY::NONE;
    // How many draws are going on inside each other
    int nesting = 0;
    tripoint_bub_ms pos;
    bool is_item_marker = false;
    // The terrain classification of the current cell
    bool terrain_classified = false;
    bool terrain_blocks = false;
    // Draw everything flat on the floor (cursors and other markers)
    bool force_floor = false;

    // Scratch memory that is reused between sprites
    render3d::mesh mesh;
    std::vector<gpu_vertex> gpu_vertices;
    // Size of the texture that was used last
    SDL_Texture *last_texture = nullptr;
    float last_texture_w = 1.0f;
    float last_texture_h = 1.0f;
};

bool view_3d_enabled()
{
    return use_tiles && get_option<bool>( "VIEW_3D" );
}

float view_3d_heading()
{
    return heading().degrees;
}

void turn_view_3d( float degrees )
{
    heading_state &h = heading();
    h.degrees = std::fmod( h.degrees + degrees + 360.0f, 360.0f );
}

std::optional<point_bub_ms> view_3d_pick( const point &pixel )
{
    const heading_state &h = heading();
    if( !h.last_camera ) {
        return std::nullopt;
    }
    const render3d::camera &cam = *h.last_camera;
    const render3d::vec3 dir = cam.pixel_direction( static_cast<float>( pixel.x ),
                               static_cast<float>( pixel.y ) );
    map &here = get_map();
    // Follow the ray until it hits the floor or the side of a block. It is cheap enough to do in
    // small steps, it happens when the pointer moves.
    constexpr float step = 0.05f;
    const float range = static_cast<float>( h.last_radius );
    for( float t = step; t <= range; t += step ) {
        const render3d::vec3 p = cam.position() + dir * t;
        const point_bub_ms cell( static_cast<int>( std::floor( p.x ) ),
                                 static_cast<int>( std::floor( p.y ) ) );
        if( p.z <= 0.0f ) {
            return cell;
        }
        if( p.z > wall_height && dir.z >= 0.0f ) {
            return std::nullopt;
        }
        if( p.z <= wall_height ) {
            const tripoint_bub_ms pos( cell, h.last_z );
            if( here.inbounds( pos ) && here.ter( pos )->movecost == 0 ) {
                return cell;
            }
        }
    }
    return std::nullopt;
}

tile_3d_scope::tile_3d_scope( scene_3d *const scene, const TILE_CATEGORY category,
                              const std::string &id, const tripoint_bub_ms &pos )
    : scene_( scene )
{
    if( !scene_ ) {
        return;
    }
    if( scene_->nesting++ > 0 ) {
        return;
    }
    scene_->category = category;
    scene_->pos = pos;
    scene_->is_item_marker = category == TILE_CATEGORY::NONE && id == "highlight_item";
    // The terrain is known from the first draw: sprites that are drawn again under another id
    // (parts of multitile sprites, the fallbacks of missing ones) are of the same terrain
    if( category == TILE_CATEGORY::TERRAIN && !scene_->terrain_classified ) {
        const ter_str_id terrain_id( id );
        scene_->terrain_blocks = terrain_id.is_valid() && terrain_id->movecost == 0;
        scene_->terrain_classified = true;
    }
}

tile_3d_scope::~tile_3d_scope()
{
    if( scene_ ) {
        scene_->nesting--;
    }
}

static placement placement_of( const scene_3d &scene )
{
    if( scene.force_floor ) {
        return placement::floor;
    }
    switch( scene.category ) {
        case TILE_CATEGORY::TERRAIN:
            return scene.terrain_blocks ? placement::block : placement::floor;
        case TILE_CATEGORY::TRAP:
        case TILE_CATEGORY::FIELD:
        case TILE_CATEGORY::LIGHTING:
            return placement::floor;
        default:
            return placement::billboard;
    }
}

// Texture coordinates of the corners of a quad, out of those of the sprite's corners. Corner
// order is top left, top right, bottom right, bottom left.
static void set_sprite_coordinates( render3d::quad &q, int quarter_turns, bool mirror )
{
    static const float base_u[4] = { 0.0f, 1.0f, 1.0f, 0.0f };
    static const float base_v[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
    static const int mirrored[4] = { 1, 0, 3, 2 };
    for( int i = 0; i < 4; ++i ) {
        int source = ( i + 4 - ( quarter_turns & 3 ) ) & 3;
        if( mirror ) {
            source = mirrored[source];
        }
        q.u[i] = base_u[source];
        q.v[i] = base_v[source];
    }
}

// Moves the coordinates of a quad, that run from 0 to 1 over the sprite, to the sprite's place in
// the atlas
static void map_to_atlas( render3d::quad &q, float u0, float v0, float u1, float v1 )
{
    for( int i = 0; i < 4; ++i ) {
        q.u[i] = u0 + q.u[i] * ( u1 - u0 );
        q.v[i] = v0 + q.v[i] * ( v1 - v0 );
    }
}

// Projects the quad and sends it to the renderer. Shading only makes sense for sprites that are
// drawn as they are: the shader variants of the GPU path use the vertex color for other things.
static void submit_quad( const SDL_Renderer_Ptr &renderer, scene_3d &scene,
                         SDL_Texture *const texture, const render3d::quad &q, const float shade,
                         const bool allow_shading )
{
    int nu = 1;
    int nv = 1;
    render3d::suggest_subdivisions( scene.cam, q, max_subdivisions, nu, nv );
    scene.mesh.clear();
    render3d::tessellate( scene.cam, q, nu, nv, scene.mesh );
    if( scene.mesh.indices.empty() ) {
        return;
    }

    const float fog_end = static_cast<float>( scene.radius );
    const float fog_start = fog_end * fog_start_fraction;
    scene.gpu_vertices.resize( scene.mesh.vertices.size() );
    for( size_t i = 0; i < scene.mesh.vertices.size(); ++i ) {
        const render3d::screen_vertex &sv = scene.mesh.vertices[i];
        float brightness = 1.0f;
        if( allow_shading ) {
            brightness = shade * render3d::distance_shade( sv.depth, fog_start, fog_end,
                         fog_min_brightness );
        }
        gpu_vertex &gv = scene.gpu_vertices[i];
        gv.x = sv.x + static_cast<float>( scene.origin.x );
        gv.y = sv.y + static_cast<float>( scene.origin.y );
        gv.color = SDL_FColor{ brightness, brightness, brightness, 1.0f };
        gv.u = sv.u;
        gv.v = sv.v;
    }
    const gpu_vertex *const first = scene.gpu_vertices.data();
    RenderGeometryTextured( renderer, texture,
                            &first->x, static_cast<int>( sizeof( gpu_vertex ) ),
                            &first->color, static_cast<int>( sizeof( gpu_vertex ) ),
                            &first->u, static_cast<int>( sizeof( gpu_vertex ) ),
                            static_cast<int>( scene.gpu_vertices.size() ),
                            scene.mesh.indices.data(), static_cast<int>( scene.mesh.indices.size() ) );
}

bool cata_tiles::draw_sprite_3d( const tile_type &tile, const texture &sprite_tex,
                                 const bool shader_bound, const bool rotate_sprite, const int rota,
                                 const point &offset )
{
    scene_3d &scene = *m_scene3d;
    SDL_Texture *const sdl_texture = sprite_tex.get_texture_ptr().get();
    if( !sdl_texture ) {
        return false;
    }
    if( sdl_texture != scene.last_texture ) {
        float w = 0.0f;
        float h = 0.0f;
        if( !SDL_GetTextureSize( sdl_texture, &w, &h ) || w <= 0.0f || h <= 0.0f ) {
            return false;
        }
        scene.last_texture = sdl_texture;
        scene.last_texture_w = w;
        scene.last_texture_h = h;
    }
    const SDL_Rect &src = sprite_tex.get_srcrect();
    const float u0 = ( static_cast<float>( src.x ) + texel_inset ) / scene.last_texture_w;
    const float v0 = ( static_cast<float>( src.y ) + texel_inset ) / scene.last_texture_h;
    const float u1 = ( static_cast<float>( src.x + src.w ) - texel_inset ) / scene.last_texture_w;
    const float v1 = ( static_cast<float>( src.y + src.h ) - texel_inset ) / scene.last_texture_h;

    // Same rules as the flat renderer: -1 mirrors the sprite, other values turn it. Isometric
    // tiles never turn.
    const bool mirror = rotate_sprite && rota == -1;
    int quarter_turns = 0;
    if( rotate_sprite && rota != -1 && !is_isometric() ) {
        quarter_turns = ( ( rota % 4 ) + 4 ) % 4;
    }

    const render3d::camera &cam = scene.cam;
    const float x = static_cast<float>( scene.pos.x() );
    const float y = static_cast<float>( scene.pos.y() );
    const bool allow_shading = !shader_bound;

    switch( placement_of( scene ) ) {
        case placement::floor: {
            const render3d::vec3 lo = { x, y, 0.0f };
            const render3d::vec3 hi = { x + 1.0f, y + 1.0f, 0.0f };
            render3d::quad q = render3d::make_box_face( lo, hi, render3d::face::top );
            if( !render3d::faces_camera( cam, lo, render3d::face_normal( render3d::face::top ) ) ) {
                break;
            }
            set_sprite_coordinates( q, quarter_turns, mirror );
            map_to_atlas( q, u0, v0, u1, v1 );
            render3d::grow_quad( cam, q, seam_overlap_pixels );
            submit_quad( renderer, scene, sdl_texture, q, shade_top, allow_shading );
            break;
        }
        case placement::block: {
            const render3d::vec3 lo = { x, y, 0.0f };
            const render3d::vec3 hi = { x + 1.0f, y + 1.0f, wall_height };
            static const std::array<render3d::face, 5> visible_faces = {
                render3d::face::north, render3d::face::south, render3d::face::east,
                render3d::face::west, render3d::face::top
            };
            for( const render3d::face f : visible_faces ) {
                render3d::quad q = render3d::make_box_face( lo, hi, f );
                if( !render3d::faces_camera( cam, q.corner[0], render3d::face_normal( f ) ) ) {
                    continue;
                }
                set_sprite_coordinates( q, 0, false );
                map_to_atlas( q, u0, v0, u1, v1 );
                render3d::grow_quad( cam, q, seam_overlap_pixels );
                submit_quad( renderer, scene, sdl_texture, q, face_shade( f ), allow_shading );
            }
            break;
        }
        case placement::billboard: {
            const float tileset_w = static_cast<float>( tileset_ptr->get_tile_width() );
            const float tileset_h = static_cast<float>( tileset_ptr->get_tile_height() );
            if( tileset_w <= 0.0f || tileset_h <= 0.0f ) {
                return false;
            }
            const render3d::vec3 anchor = { x + 0.5f, y + 0.5f, 0.0f };
            if( std::hypot( anchor.x - cam.position().x, anchor.y - cam.position().y ) <
                min_billboard_distance ) {
                break;
            }
            // The sprite covers the tile it is drawn on the way the flat renderer places it: its
            // size and offset are in pixels of the tileset, where a tile is tileset_w by tileset_h.
            const point sprite_offset = tile.offset + offset;
            const float sprite_w = static_cast<float>( src.w ) * tile.pixelscale;
            const float sprite_h = static_cast<float>( src.h ) * tile.pixelscale;
            const float left = static_cast<float>( sprite_offset.x ) / tileset_w - 0.5f;
            const float right = left + sprite_w / tileset_w;
            const float bottom = 1.0f - ( static_cast<float>( sprite_offset.y ) + sprite_h ) / tileset_h;
            const float top = bottom + sprite_h / tileset_h;
            const float scale = scene.category == TILE_CATEGORY::ITEM || scene.is_item_marker ?
                                item_scale : 1.0f;
            render3d::quad q = render3d::make_billboard( cam, anchor, left * scale, right * scale,
                               bottom * scale, top * scale, mirror );
            map_to_atlas( q, u0, v0, u1, v1 );
            // Every vertex is as far from the camera as the middle of the sprite
            submit_quad( renderer, scene, sdl_texture, q, 1.0f, allow_shading );
            break;
        }
    }
    return true;
}

void cata_tiles::draw_3d( const point &dest, const tripoint_bub_ms &center, const int width,
                          const int height )
{
    display_buffer_draw_scope draw_scope;
    if( display_buffer_scope_is_invalid() || !g || width <= 0 || height <= 0 ) {
        return;
    }

    has_animated_tiles_ = false;

    {
        // Keep everything inside the terrain window, and start from black
        SDL_Rect clip_rect = { dest.x, dest.y, width, height };
        RenderSetClipRect( renderer, &clip_rect );
        flush_sprite_shader_for_untextured_draw();
        geometry->rect( renderer, clip_rect, SDL_Color{ 0, 0, 0, 255 } );
    }

    init_light();
    map &here = get_map();
    avatar &you = get_avatar();
    creature_tracker &creatures = get_creature_tracker();
    const visibility_variables &cache = here.get_visibility_variables_cache();
    const level_cache &ch = here.access_cache( center.z() );

    follow_avatar( you.pos_abs() );

    scene_3d scene;
    scene.origin = dest;
    scene.radius = std::clamp( get_option<int>( "VIEW_3D_DISTANCE" ), 4, 60 );
    scene.cam = render3d::camera(
                    render3d::vec3{ static_cast<float>( center.x() ) + 0.5f,
                                    static_cast<float>( center.y() ) + 0.5f, eye_height },
                    heading().degrees, 0.0f,
                    static_cast<float>( get_option<int>( "VIEW_3D_FOV" ) ), width, height );

    heading().last_camera = scene.cam;
    heading().last_z = center.z();
    heading().last_radius = scene.radius;

    // Sprites are placed in the scene, and never retracted to see behind them
    const bool previous_disable_occlusion = disable_occlusion;
    disable_occlusion = true;
    m_scene3d = &scene;
    on_out_of_scope clear_scene( [&]() {
        m_scene3d = nullptr;
        disable_occlusion = previous_disable_occlusion;
    } );

    // What the character can see is a square around them, everything else can only be remembered
    const tripoint_bub_ms you_pos = you.pos_bub( here );
    const point min_visible( you_pos.x() % SEEX, you_pos.y() % SEEY );
    const point max_visible( ( you_pos.x() % SEEX ) + ( MAPSIZE - 1 ) * SEEX,
                             ( you_pos.y() % SEEY ) + ( MAPSIZE - 1 ) * SEEY );
    // Map memory has to cover what is drawn, and what is seen
    const point min_memory( std::min( min_visible.x, center.x() - scene.radius ),
                            std::min( min_visible.y, center.y() - scene.radius ) );
    const point max_memory( std::max( max_visible.x, center.x() + scene.radius ),
                            std::max( max_visible.y, center.y() + scene.radius ) );
    you.prepare_map_memory_region(
        here.get_abs( tripoint_bub_ms( min_memory.x, min_memory.y, center.z() ) ),
        here.get_abs( tripoint_bub_ms( max_memory.x, max_memory.y, center.z() ) ) );

    const auto vision_cache = you.get_vision_modes();
    nv_goggles_activated = vision_cache[NV_GOGGLES];

    const auto apply_visible = [&]( const tripoint_bub_ms & np ) {
        return np.y() < min_visible.y || np.y() > max_visible.y ||
               np.x() < min_visible.x || np.x() > max_visible.x ||
               would_apply_vision_effects( here.get_visibility( ch.visibility_cache[np.x()][np.y()],
                                           cache ) );
    };

    // The layers of a cell, in the order they are drawn. The ones that are user interface on top
    // of the flat map (zone marks, revival indicators) have no place here.
    const std::array<decltype( &cata_tiles::draw_furniture ), 9> layers = {{
            &cata_tiles::draw_terrain, &cata_tiles::draw_furniture, &cata_tiles::draw_graffiti,
            &cata_tiles::draw_trap, &cata_tiles::draw_part_con, &cata_tiles::draw_field_or_item,
            &cata_tiles::draw_vpart_no_roof, &cata_tiles::draw_vpart_roof,
            &cata_tiles::draw_critter_at
        }
    };

    int cells_drawn = 0;
    for( const render3d::cell_offset &off : render3d::far_to_near_offsets( scene.radius ) ) {
        if( ++cells_drawn % 64 == 0 && renderer_should_abort_frame() ) {
            // The renderer is going away; the next frame draws it all again
            break;
        }
        const tripoint_bub_ms pos( center.x() + off.dx, center.y() + off.dy, center.z() );
        // Cells that are (nearly) outside the field of view need no work at all
        const render3d::vec3 cell_center = { static_cast<float>( pos.x() ) + 0.5f,
                                             static_cast<float>( pos.y() ) + 0.5f, 0.0f
                                           };
        if( !render3d::may_be_visible( scene.cam, cell_center, 1.5f ) ) {
            continue;
        }

        // Same decisions as the flat renderer takes when it collects the tiles to draw: what
        // is out of sight is remembered or blank, and what is in sight is drawn with its light.
        const tripoint_abs_ms pos_global = here.get_abs( pos );
        std::array<bool, 5> invisible;
        invisible[0] = false;
        lit_level ll = lit_level::DARK;
        const bool in_sight_range = pos.x() >= min_visible.x && pos.x() <= max_visible.x &&
                                    pos.y() >= min_visible.y && pos.y() <= max_visible.y;
        if( !in_sight_range ) {
            if( you.has_memory_at( pos_global ) ) {
                ll = lit_level::MEMORIZED;
                invisible[0] = true;
            } else if( has_draw_override( pos ) ) {
                ll = lit_level::DARK;
                invisible[0] = true;
            } else {
                continue;
            }
        } else {
            ll = ch.visibility_cache[pos.x()][pos.y()];
        }
        if( !invisible[0] ) {
            const visibility_type vis_type = here.get_visibility( ll, cache );
            if( would_apply_vision_effects( vis_type ) ) {
                const Creature *critter = creatures.creature_at( pos, true );
                if( has_draw_override( pos ) || you.has_memory_at( pos_global ) ||
                    ( critter &&
                      ( critter->has_flag( mon_flag_ALWAYS_VISIBLE )
                        || you.sees_with_specials( *critter ) ) ) ) {
                    invisible[0] = true;
                } else {
                    continue;
                }
            }
        }
        for( int i = 0; i < 4; i++ ) {
            invisible[1 + i] = apply_visible( pos + neighborhood[i] );
        }

        scene.terrain_classified = false;
        const bool camera_cell = off.dx == 0 && off.dy == 0;
        int height_3d = 0;
        for( const auto layer : layers ) {
            // The camera is the character's eyes, it does not see itself
            if( camera_cell && layer == &cata_tiles::draw_critter_at ) {
                continue;
            }
            ( this->*layer )( pos, ll, height_3d, invisible, false );
        }
    }

    // Cursors, highlights and lines are markers on the floor. They are drawn last, so they show
    // through walls.
    scene.force_floor = true;
    if( do_draw_line ) {
        draw_line();
    }
    if( do_draw_cursor ) {
        draw_cursor();
    }
    if( do_draw_highlight ) {
        draw_highlight();
    }
    scene.force_floor = false;

    // Animations of the flat renderer (bullets, explosions, weather, floating text) are drawn at
    // places on the screen that mean nothing here; forget about the ones that live for a single
    // frame like the flat renderer does.
    void_line();
    void_cursor();
    void_highlight();
    void_weather();
    void_sct();
    void_zones();
    void_hit();
    in_animation = false;

    // tile overrides are already drawn in the previous code
    void_radiation_override();
    void_terrain_override();
    void_furniture_override();
    void_graffiti_override();
    void_trap_override();
    void_field_override();
    void_item_override();
    void_vpart_override();
    void_monster_override();

    // Map memory does not depend on what is on the screen
    memorize_visible_map( min_visible, max_visible, center.z(), {} );

    RenderSetClipRect( renderer, nullptr );
    // Unbind any GPU render state held across sprite batches so ImGui or the
    // next-frame draws see clean state. On flush failure the bind boundary
    // forbids a target switch: abort the unbind, latch recovery, and throw.
    if( cata_shader::variant_pass *vp = get_shared_variant_pass() ) {
        if( !vp->flush() ) {
            draw_scope.abort_unbind();
            display_buffer_scope_signal_recovery_required();
            throw std::runtime_error(
                "cata_tiles::draw_3d: variant_pass flush failed at end of frame; renderer in undefined state" );
        }
    }
}

#endif // TILES
