#if defined(TILES)

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "calendar.h"
#include "cata_catch.h"
#include "cata_tiles.h"
#include "character.h"
#include "coordinates.h"
#include "game.h"
#include "item.h"
#include "map.h"
#include "map_helpers.h"
#include "map_helpers_tests.h"
#include "player_helpers.h"
#include "point.h"
#include "sdl_geometry.h"
#include "sdl_renderer_recovery.h"
#include "sdl_wrappers.h"
#include "sdltiles.h"
#include "type_id.h"

static const itype_id itype_2x4( "2x4" );

static const ter_str_id ter_t_door_c( "t_door_c" );
static const ter_str_id ter_t_floor( "t_floor" );
static const ter_str_id ter_t_grass( "t_grass" );
static const ter_str_id ter_t_tree( "t_tree" );
static const ter_str_id ter_t_wall( "t_wall" );
static const ter_str_id ter_t_window( "t_window" );

namespace
{

constexpr int view_w = 640;
constexpr int view_h = 360;
// Where the horizon is on the screen: the camera does not pitch
constexpr int horizon_y = view_h / 2;

// A street in front of a small building. The character stands south of it, facing north.
// The map only stays where it is while the character is in x and y from 60 to 71, that is why
// the whole scene is down here.
//
//   y=60   ###########
//   y=63   #  T  Z   #       walls, a table (T) and a zombie (Z) inside
//   y=66   ###WD#W####        a door (D) and windows (W) in the front wall
//   y=68       Z   t         another zombie and a tree outside
//   y=69      i              an item
//   y=71        @
struct view_3d_fixture {
    view_3d_fixture() : available_( fx_.available() ) {
        if( !available_ ) {
            return;
        }
        renderer_recovery_test_support::set_scaling_and_resize_window( 1, view_w, view_h );
        renderer_coordinator.drain_pending();
        int out_w = 0;
        int out_h = 0;
        GetRendererOutputSize( get_sdl_renderer(), &out_w, &out_h );
        int buffer_w = 0;
        int buffer_h = 0;
        renderer_recovery_test_support::current_display_buffer_dims( buffer_w, buffer_h );
        available_ = out_w >= view_w && out_h >= view_h && buffer_w >= view_w && buffer_h >= view_h;
        if( !available_ ) {
            return;
        }

        tiles_ = std::make_unique<cata_tiles>( get_sdl_renderer(), geometry_, cache_ );
        const char *const tileset = std::getenv( "CATA_VIEW_3D_TILESET" );
        try {
            tiles_->load_tileset( tileset ? tileset : "ASCIITiles", false, true, false, false );
        } catch( const std::exception & ) {
            available_ = false;
            return;
        }
        available_ = tiles_->is_valid();
        if( !available_ ) {
            return;
        }

        clear_avatar();
        clear_map();
        build_test_map( ter_t_grass.id() );
        set_time_to_day();
        build_scene();
    }

    ~view_3d_fixture() {
        // The textures must go while the renderer they belong to is still there
        tiles_.reset();
        cache_.release_live_atlases();
    }

    bool available() const {
        return available_;
    }

    static void refresh_map_caches() {
        map &here = get_map();
        here.invalidate_map_cache( 0 );
        here.build_map_cache( 0, true );
        here.invalidate_visibility_cache();
        here.update_visibility_cache( 0 );
    }

    void build_scene() {
        map &here = get_map();
        for( int x = 60; x <= 70; ++x ) {
            for( int y = 60; y <= 66; ++y ) {
                const bool border = x == 60 || x == 70 || y == 60 || y == 66;
                here.ter_set( tripoint_bub_ms( x, y, 0 ), border ? ter_t_wall : ter_t_floor );
            }
        }
        here.ter_set( tripoint_bub_ms( 65, 66, 0 ), ter_t_door_c );
        here.ter_set( tripoint_bub_ms( 62, 66, 0 ), ter_t_window );
        here.ter_set( tripoint_bub_ms( 68, 66, 0 ), ter_t_window );
        here.ter_set( tripoint_bub_ms( 61, 69, 0 ), ter_t_tree );
        here.ter_set( tripoint_bub_ms( 71, 68, 0 ), ter_t_tree );
        here.furn_set( tripoint_bub_ms( 63, 63, 0 ), furn_id( "f_table" ) );
        here.add_item( tripoint_bub_ms( 64, 69, 0 ), item( itype_2x4, calendar::turn ) );
        spawn_test_monster( "mon_zombie", tripoint_bub_ms( 66, 63, 0 ), false );
        spawn_test_monster( "mon_zombie", tripoint_bub_ms( 67, 68, 0 ), false );
        g->place_player( tripoint_bub_ms( 65, 71, 0 ) );
        refresh_map_caches();
    }

    // Draws a frame, looking the way the character is looking, from the character's tile
    void draw() {
        // The first frame lets the view find out where the character is
        const tripoint_bub_ms center = get_player_character().pos_bub();
        tiles_->draw_3d( point::zero, center, view_w, view_h );
    }

    void look( float heading_degrees ) {
        turn_view_3d( heading_degrees - view_3d_heading() );
    }

    // Draws a frame looking the given way and reads the picture back
    void draw_facing( float heading_degrees ) {
        draw();
        look( heading_degrees );
        draw();
    }

    // All the pixels of the last frame, as 0xAARRGGBB
    std::vector<Uint32> pixels() const {
        display_buffer_draw_scope scope;
        REQUIRE( scope.should_draw() );
        std::vector<Uint32> out( static_cast<size_t>( view_w ) * view_h, 0 );
        const SDL_Rect rect = { 0, 0, view_w, view_h };
        REQUIRE( RenderReadPixels( get_sdl_renderer(), &rect, SDL_PIXELFORMAT_ARGB8888, out.data(),
                                   view_w * static_cast<int>( sizeof( Uint32 ) ) ) );
        return out;
    }

    // How much of the rectangle is not black, from 0 to 1
    static double coverage( const std::vector<Uint32> &image, const SDL_Rect &rect ) {
        int lit = 0;
        for( int y = rect.y; y < rect.y + rect.h; ++y ) {
            for( int x = rect.x; x < rect.x + rect.w; ++x ) {
                if( ( image[static_cast<size_t>( y ) * view_w + x] & 0x00FFFFFF ) != 0 ) {
                    ++lit;
                }
            }
        }
        return static_cast<double>( lit ) / static_cast<double>( rect.w * rect.h );
    }

    // Writes the last frame as a binary PPM when CATA_VIEW_3D_SCREENSHOT_DIR is set
    void screenshot( const std::string &name ) const {
        const char *const dir = std::getenv( "CATA_VIEW_3D_SCREENSHOT_DIR" );
        if( !dir ) {
            return;
        }
        const std::vector<Uint32> image = pixels();
        std::ofstream out( std::string( dir ) + "/" + name + ".ppm", std::ios::binary );
        out << "P6\n" << view_w << " " << view_h << "\n255\n";
        for( const Uint32 pixel : image ) {
            const char rgb[3] = { static_cast<char>( ( pixel >> 16 ) & 0xFF ),
                                  static_cast<char>( ( pixel >> 8 ) & 0xFF ),
                                  static_cast<char>( pixel & 0xFF )
                                };
            out.write( rgb, 3 );
        }
    }

    software_render_fixture fx_;
    GeometryRenderer_Ptr geometry_ = std::make_unique<DefaultGeometryRenderer>();
    tileset_cache cache_;
    std::unique_ptr<cata_tiles> tiles_;
    bool available_ = false;
};

} // namespace

TEST_CASE( "view_3d_draws_walls_creatures_and_floor", "[tiles][view_3d]" )
{
    view_3d_fixture fx;
    if( !fx.available() ) {
        WARN( "dummy SDL video backend or tileset unavailable; skipping" );
        return;
    }
    fx.draw_facing( 0.0f );
    fx.screenshot( "north" );
    const std::vector<Uint32> image = fx.pixels();

    SECTION( "the sky is empty" ) {
        // Nothing in the scene is taller than a block, and a block is 0.4 tiles above the eye
        CHECK( view_3d_fixture::coverage( image, SDL_Rect{ 0, 0, view_w, horizon_y - 100 } ) == 0.0 );
    }
    SECTION( "the front of the building stands at the horizon" ) {
        // Four and a half tiles away and a tile high: about 70 pixels, mostly below the horizon
        CHECK( view_3d_fixture::coverage( image, SDL_Rect{ 250, horizon_y - 8, 140, 16 } ) > 0.05 );
    }
    SECTION( "the ground is drawn" ) {
        CHECK( view_3d_fixture::coverage( image, SDL_Rect{ 0, horizon_y + 60, view_w, view_h - horizon_y - 60 } )
               > 0.02 );
    }

    SECTION( "there is nothing at the horizon behind the character" ) {
        fx.look( 180.0f );
        fx.draw();
        const std::vector<Uint32> behind = fx.pixels();
        fx.screenshot( "south" );
        const SDL_Rect band = { 250, horizon_y - 8, 140, 16 };
        CHECK( view_3d_fixture::coverage( behind, band ) < view_3d_fixture::coverage( image, band ) / 2 );
    }
}

TEST_CASE( "view_3d_follows_the_character", "[tiles][view_3d]" )
{
    view_3d_fixture fx;
    if( !fx.available() ) {
        WARN( "dummy SDL video backend or tileset unavailable; skipping" );
        return;
    }
    fx.draw();

    SECTION( "stepping east turns the view east" ) {
        g->place_player( tripoint_bub_ms( 66, 71, 0 ) );
        view_3d_fixture::refresh_map_caches();
        fx.draw();
        CHECK( view_3d_heading() == Approx( 90.0f ) );
    }
    SECTION( "stepping north west turns the view north west" ) {
        g->place_player( tripoint_bub_ms( 64, 70, 0 ) );
        view_3d_fixture::refresh_map_caches();
        fx.draw();
        CHECK( view_3d_heading() == Approx( 315.0f ) );
    }
    SECTION( "jumping far away does not turn the view" ) {
        fx.look( 0.0f );
        g->place_player( tripoint_bub_ms( 70, 62, 0 ) );
        view_3d_fixture::refresh_map_caches();
        fx.draw();
        CHECK( view_3d_heading() == Approx( 0.0f ) );
    }
    SECTION( "turning adds up and wraps around" ) {
        fx.look( 0.0f );
        turn_view_3d( 45.0f );
        CHECK( view_3d_heading() == Approx( 45.0f ) );
        turn_view_3d( -90.0f );
        CHECK( view_3d_heading() == Approx( 315.0f ) );
        turn_view_3d( 90.0f );
        CHECK( view_3d_heading() == Approx( 45.0f ) );
    }
}

TEST_CASE( "view_3d_picks_what_is_under_the_pointer", "[tiles][view_3d]" )
{
    view_3d_fixture fx;
    if( !fx.available() ) {
        WARN( "dummy SDL video backend or tileset unavailable; skipping" );
        return;
    }
    fx.draw_facing( 0.0f );

    SECTION( "the door in the middle of the horizon" ) {
        const std::optional<point_bub_ms> cell = view_3d_pick( point( view_w / 2, horizon_y ) );
        REQUIRE( cell.has_value() );
        CHECK( *cell == point_bub_ms( 65, 66 ) );
    }
    SECTION( "the wall left of the door" ) {
        // The wall is 4.5 tiles away: 3 tiles left of the door is at x = 320 - 3 / 4.5 * 320
        const std::optional<point_bub_ms> cell = view_3d_pick( point( view_w / 2 - 213, horizon_y ) );
        REQUIRE( cell.has_value() );
        CHECK( cell->y() == 66 );
        CHECK( cell->x() == 62 );
    }
    SECTION( "the floor just in front of the character" ) {
        const std::optional<point_bub_ms> cell = view_3d_pick( point( view_w / 2, view_h - 1 ) );
        REQUIRE( cell.has_value() );
        CHECK( cell->x() == 65 );
        CHECK( cell->y() >= 69 );
        CHECK( cell->y() <= 71 );
    }
    SECTION( "the sky is nothing" ) {
        CHECK_FALSE( view_3d_pick( point( view_w / 2, 10 ) ).has_value() );
    }
    SECTION( "the far side of the world is picked when turned around" ) {
        fx.look( 90.0f );
        fx.draw();
        const std::optional<point_bub_ms> cell = view_3d_pick( point( view_w / 2, view_h - 1 ) );
        REQUIRE( cell.has_value() );
        CHECK( cell->y() == 71 );
        CHECK( cell->x() >= 65 );
    }
}

TEST_CASE( "view_3d_writes_pictures", "[tiles][view_3d][.]" )
{
    // Not a test: writes pictures of the scene from a few places when asked to
    if( !std::getenv( "CATA_VIEW_3D_SCREENSHOT_DIR" ) ) {
        return;
    }
    view_3d_fixture fx;
    if( !fx.available() ) {
        WARN( "dummy SDL video backend or tileset unavailable; skipping" );
        return;
    }
    const struct {
        const char *name;
        tripoint_bub_ms where;
        float heading;
    } views[] = {
        { "street_north", tripoint_bub_ms( 65, 71, 0 ), 0.0f },
        { "street_northeast", tripoint_bub_ms( 65, 71, 0 ), 45.0f },
        { "street_east", tripoint_bub_ms( 65, 71, 0 ), 90.0f },
        { "corner", tripoint_bub_ms( 70, 71, 0 ), 335.0f },
        { "wall", tripoint_bub_ms( 63, 69, 0 ), 15.0f },
        { "window", tripoint_bub_ms( 62, 69, 0 ), 0.0f },
        { "inside", tripoint_bub_ms( 64, 63, 0 ), 20.0f },
    };
    for( const auto &view : views ) {
        g->place_player( view.where );
        view_3d_fixture::refresh_map_caches();
        fx.draw_facing( view.heading );
        fx.screenshot( view.name );
    }
}

#endif // TILES
