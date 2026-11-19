#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tgaimage.h"

namespace {
constexpr int width = 448;
constexpr int height = 320;
constexpr double focal = 398.0;
constexpr double cx = (width - 1) * 0.5;
constexpr double cy = (height - 1) * 0.5;
constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double pi = 3.141592653589793;
constexpr double depth_gate = 0.04;

struct Vec {
    double x = 0, y = 0, z = 0;
    Vec operator+(Vec b) const { return {x + b.x, y + b.y, z + b.z}; }
    Vec operator-(Vec b) const { return {x - b.x, y - b.y, z - b.z}; }
    Vec operator*(double factor) const { return {x * factor, y * factor, z * factor}; }
};

double dot(Vec a, Vec b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

struct Intrinsics {
    int cols = width, rows = height;
    double fx = focal, fy = focal, u0 = cx, v0 = cy;

    Vec ray(int x, int y) const { return {(x - u0) / fx, (y - v0) / fy, 1}; }
};

struct Camera {
    Vec center;
    std::array<Vec, 3> axes;  // Columns of the camera-to-world rotation.

    static Camera planar(double shift, double yaw) {
        const double c = std::cos(yaw * pi / 180), s = std::sin(yaw * pi / 180);
        return {{shift, 0, 0}, {{{c, 0, -s}, {0, 1, 0}, {s, 0, c}}}};
    }

    Vec rotation(Vec p) const { return axes[0] * p.x + axes[1] * p.y + axes[2] * p.z; }
    Vec to_camera(Vec p) const {
        const Vec d = p - center;
        return {dot(axes[0], d), dot(axes[1], d), dot(axes[2], d)};
    }
};

struct Sphere { Vec center; double radius; Vec color; };
using Scene = std::array<Sphere, 3>;

struct Pixel { Vec rgb; double depth = infinity; };
using Frame = std::vector<Pixel>;

Pixel trace(const Camera& camera, Vec direction, const Scene& scene) {
    // Rays use z = 1 in camera coordinates, so t is camera-axis depth.
    const double plane_z = 8.2;
    const double plane_t = (plane_z - camera.center.z) / direction.z;
    const Vec wall = camera.center + direction * plane_t;
    const double floor_x = std::floor(wall.x * 2.2);
    const double floor_y = std::floor(wall.y * 2.2);
    const bool checker = std::fmod(std::abs(floor_x + floor_y), 2.0) > 0.5;
    const bool seam = wall.x * 2.2 - floor_x < 0.055 || wall.y * 2.2 - floor_y < 0.055;
    Pixel hit{seam ? Vec{76, 104, 112} : checker ? Vec{186, 209, 212} : Vec{218, 228, 220}, plane_t};
    const double a = dot(direction, direction);

    for (const Sphere& sphere : scene) {
        const Vec delta = camera.center - sphere.center;
        const double b = dot(delta, direction);
        const double discriminant = b * b - a * (dot(delta, delta) - sphere.radius * sphere.radius);
        if (discriminant < 0) continue;
        const double t = (-b - std::sqrt(discriminant)) / a;
        if (t <= 0 || t >= hit.depth) continue;
        const Vec position = camera.center + direction * t;
        const Vec normal = (position - sphere.center) * (1 / sphere.radius);
        const double shade = 0.38 + 0.62 * std::max(0.0, dot(normal, {-0.36, -0.54, -0.76}));
        hit = {sphere.color * shade, t};
    }
    return hit;
}

Frame render(const Camera& camera, const Scene& scene) {
    const Intrinsics k;
    Frame image(width * height);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            image[y * width + x] = trace(camera, camera.rotation(k.ray(x, y)), scene);
    return image;
}

struct Sample {
    double x = 0, y = 0, z = infinity;
    Vec rgb;
};

Sample project_pixel(const Pixel& pixel, Vec ray, const Camera& source, const Camera& target,
                     const Intrinsics& target_k) {
    if (!std::isfinite(pixel.depth) || pixel.depth <= 0) return {};
    const Vec world = source.center + source.rotation(ray * pixel.depth);
    const Vec p = target.to_camera(world);
    if (p.z <= 1e-6) return {};
    return {target_k.fx * p.x / p.z + target_k.u0, target_k.fy * p.y / p.z + target_k.v0,
            p.z, pixel.rgb};
}

bool reaches_image(const Sample& s, const Intrinsics& k) {
    return std::isfinite(s.z) && s.x > -1 && s.x < k.cols && s.y > -1 && s.y < k.rows;
}

std::vector<Sample> reproject(const Frame& frame, const Camera& source, const Camera& target, const Intrinsics& k = {}) {
    std::vector<Sample> samples;
    samples.reserve(frame.size());
    for (int y = 0; y < k.rows; ++y) {
        for (int x = 0; x < k.cols; ++x) {
            const Sample s = project_pixel(frame[y * k.cols + x], k.ray(x, y), source, target, k);
            if (reaches_image(s, k)) samples.push_back(s);
        }
    }
    return samples;
}

template<class Write>
void splat(const Sample& s, const Intrinsics& k, Write write, double radius = 1) {
    if (radius > 1) {
        const int xmin = static_cast<int>(std::max(0.0, std::ceil(s.x - radius)));
        const int xmax = static_cast<int>(std::min(k.cols - 1.0, std::floor(s.x + radius)));
        const int ymin = static_cast<int>(std::max(0.0, std::ceil(s.y - radius)));
        const int ymax = static_cast<int>(std::min(k.rows - 1.0, std::floor(s.y + radius)));
        for (int y = ymin; y <= ymax; ++y) {
            for (int x = xmin; x <= xmax; ++x) {
                const double weight = (1 - std::abs(s.x - x) / radius) * (1 - std::abs(s.y - y) / radius);
                if (weight > 1e-7) write(y * k.cols + x, weight);
            }
        }
        return;
    }
    const int left = static_cast<int>(std::floor(s.x)), top = static_cast<int>(std::floor(s.y));
    const double fx = s.x - left, fy = s.y - top;
    for (int dy = 0; dy <= 1; ++dy) {
        for (int dx = 0; dx <= 1; ++dx) {
            const int x = left + dx, y = top + dy;
            const double weight = (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy);
            if (x >= 0 && y >= 0 && x < k.cols && y < k.rows && weight > 1e-7)
                write(y * k.cols + x, weight);
        }
    }
}

Frame nearest(const std::vector<Sample>& samples) {
    Frame frame(width * height);
    for (const Sample& s : samples) {
        const int x = static_cast<int>(std::lround(s.x)), y = static_cast<int>(std::lround(s.y));
        if (x < 0 || y < 0 || x >= width || y >= height) continue;
        Pixel& pixel = frame[y * width + x];
        if (s.z < pixel.depth) pixel = {s.rgb, s.z};
    }
    return frame;
}

Frame bilinear(const std::vector<Sample>& samples, double gate, const Intrinsics& k = {}, double radius = 1) {
    Frame frame(k.cols * k.rows);
    std::vector<double> weights(frame.size(), 0);
    for (const Sample& s : samples)
        splat(s, k, [&](int i, double) { frame[i].depth = std::min(frame[i].depth, s.z); }, radius);
    for (const Sample& s : samples) {
        splat(s, k, [&](int i, double w) {
            if (s.z - frame[i].depth > gate) return;
            frame[i].rgb = frame[i].rgb + s.rgb * w;
            weights[i] += w;
        }, radius);
    }
    for (std::size_t i = 0; i < frame.size(); ++i)
        if (weights[i] > 0) frame[i].rgb = frame[i].rgb * (1 / weights[i]);
    return frame;
}

bool valid(const Pixel& pixel) { return std::isfinite(pixel.depth); }

double psnr(double sqerr, std::size_t pixels) {
    if (pixels == 0) return std::numeric_limits<double>::quiet_NaN();
    if (sqerr < 1e-10) return infinity;
    return 10 * std::log10(255.0 * 255.0 * 3.0 * pixels / sqerr);
}

unsigned char byte(double value) {
    return static_cast<unsigned char>(std::lround(std::clamp(value, 0.0, 255.0)));
}

Vec rounded(Vec c) { return {double(byte(c.x)), double(byte(c.y)), double(byte(c.z))}; }

struct Metrics { double holes, all_psnr, common_psnr; };

Metrics evaluate(const Frame& predicted, const Frame& reference, const Frame& nearest_frame, const Frame& gated_frame) {
    std::size_t invalid = 0, common = 0;
    double full_error = 0, common_error = 0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        if (!valid(predicted[i])) ++invalid;
        const Vec delta = rounded(predicted[i].rgb) - rounded(reference[i].rgb);
        const double error = dot(delta, delta);
        full_error += error;
        if (!valid(nearest_frame[i]) || !valid(gated_frame[i])) continue;
        ++common;
        common_error += error;
    }
    return {100.0 * invalid / reference.size(), psnr(full_error, reference.size()), psnr(common_error, common)};
}

void write_frame(const std::filesystem::path& path, const Frame& frame, const Intrinsics& k = {}) {
    TGAImage image(k.cols, k.rows, TGAImage::RGB);
    for (int y = 0; y < k.rows; ++y) {
        for (int x = 0; x < k.cols; ++x) {
            const Vec color = frame[y * k.cols + x].rgb;
            image.set(x, y, {{byte(color.z), byte(color.y), byte(color.x), 255}, 3});
        }
    }
    if (!image.write_tga_file(path.string(), false, false)) throw std::runtime_error("Could not write " + path.string());
}


Frame read_rgbd(const std::filesystem::path& path, const Intrinsics& k) {
    std::ifstream file(path, std::ios::binary);
    std::vector<unsigned char> bytes(static_cast<std::size_t>(k.cols) * k.rows * 5);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), bytes.size()) ||
        file.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Incorrect RGB-D byte count: " + path.string());

    Frame frame(bytes.size() / 5);
    for (std::size_t i = 0; i < frame.size(); ++i) {
        const auto* p = &bytes[5 * i];
        const int depth_mm = p[3] + 256 * p[4];
        frame[i] = {{double(p[0]), double(p[1]), double(p[2])},
                    depth_mm > 0 ? depth_mm / 1000.0 : infinity};
    }
    return frame;
}


Camera read_camera(std::istream& input, double r00) {
    Camera camera{};
    camera.axes[0].x = r00;
    if (!(input >> camera.axes[1].x >> camera.axes[2].x >> camera.center.x
                >> camera.axes[0].y >> camera.axes[1].y >> camera.axes[2].y >> camera.center.y
                >> camera.axes[0].z >> camera.axes[1].z >> camera.axes[2].z >> camera.center.z))
        throw std::runtime_error("Incomplete camera-to-world transform");
    return camera;
}

double edge(const Sample& a, const Sample& b, double x, double y) {
    return (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x);
}

void rasterize(const Sample& a, const Sample& b, const Sample& c, const Intrinsics& k, Frame& frame) {
    const double area = edge(a, b, c.x, c.y);
    if (std::abs(area) < 1e-8) return;
    const double left = std::max(0.0, std::ceil(std::min({a.x, b.x, c.x})));
    const double right = std::min(k.cols - 1.0, std::floor(std::max({a.x, b.x, c.x})));
    const double top = std::max(0.0, std::ceil(std::min({a.y, b.y, c.y})));
    const double bottom = std::min(k.rows - 1.0, std::floor(std::max({a.y, b.y, c.y})));
    if (left > right || top > bottom) return;
    for (int y = static_cast<int>(top); y <= static_cast<int>(bottom); ++y) {
        for (int x = static_cast<int>(left); x <= static_cast<int>(right); ++x) {
            const double wa = edge(b, c, x, y) / area;
            const double wb = edge(c, a, x, y) / area;
            const double wc = 1 - wa - wb;
            if (std::min({wa, wb, wc}) < -1e-8) continue;
            const double ta = wa / a.z, tb = wb / b.z, tc = wc / c.z;
            const double z = 1 / (ta + tb + tc);
            Pixel& dst = frame[y * k.cols + x];
            if (z < dst.depth) dst = {(a.rgb * ta + b.rgb * tb + c.rgb * tc) * z, z};
        }
    }
}

double surface_tolerance(double z) { return 0.012 + 0.035 * z; }

bool continuous(double a, double b, double c) {
    const double nearest = std::min({a, b, c});
    return std::max({a, b, c}) - nearest <= surface_tolerance(nearest);
}

Frame mesh_render(const Frame& source, const std::vector<Sample>& projected, const Intrinsics& source_k,
                  const Intrinsics& target_k, bool gate) {
    Frame rendered(target_k.cols * target_k.rows);
    const auto triangle = [&](int i, int j, int k) {
        if (!std::isfinite(projected[i].z) || !std::isfinite(projected[j].z) ||
            !std::isfinite(projected[k].z)) return;
        if (gate && !continuous(source[i].depth, source[j].depth, source[k].depth)) return;
        rasterize(projected[i], projected[j], projected[k], target_k, rendered);
    };
    for (int y = 0; y + 1 < source_k.rows; ++y) {
        for (int x = 0; x + 1 < source_k.cols; ++x) {
            const int a = y * source_k.cols + x, b = a + 1, c = a + source_k.cols, d = c + 1;
            const double ad = std::abs(source[a].depth - source[d].depth);
            const double bc = std::abs(source[b].depth - source[c].depth);
            // A diagonal with an invalid endpoint cannot support either triangle.
            if (std::isfinite(ad) && (!std::isfinite(bc) || ad <= bc)) {
                triangle(a, b, d);
                triangle(a, d, c);
            } else {
                triangle(a, b, c);
                triangle(b, d, c);
            }
        }
    }
    return rendered;
}


Frame complete_depth(const Frame& source, const Intrinsics& k, int radius) {
    Frame completed = source;
    constexpr double color_limit = 3 * 24.0 * 24.0;  // RGB squared distance.
    for (int y = 0; y < k.rows; ++y) {
        for (int x = 0; x < k.cols; ++x) {
            const int index = y * k.cols + x;
            const Pixel& hole = source[index];
            if (valid(hole)) continue;
            const int x0 = std::max(0, x - radius), x1 = std::min(k.cols - 1, x + radius);
            const int y0 = std::max(0, y - radius), y1 = std::min(k.rows - 1, y + radius);
            double seed = infinity, best_cost = infinity;
            for (int ny = y0; ny <= y1; ++ny) {
                for (int nx = x0; nx <= x1; ++nx) {
                    const Pixel& sample = source[ny * k.cols + nx];
                    if (!valid(sample)) continue;
                    const Vec delta = sample.rgb - hole.rgb;
                    const double color_distance = dot(delta, delta);
                    if (color_distance > color_limit) continue;
                    const int dx = nx - x, dy = ny - y;
                    const double cost = dx * dx + dy * dy + 2 * color_distance / color_limit;
                    if (cost < best_cost) {
                        best_cost = cost;
                        seed = sample.depth;
                    }
                }
            }
            if (!std::isfinite(seed)) continue;

            // Fit inverse depth to a plane using only measured samples from a compatible surface.
            double weight_sum = 0, xx = 0, yy = 0, zz = 0, xy = 0, xz = 0, yz = 0;
            Vec mean;
            int support = 0;
            for (int ny = y0; ny <= y1; ++ny) {
                for (int nx = x0; nx <= x1; ++nx) {
                    const Pixel& sample = source[ny * k.cols + nx];
                    if (!valid(sample) || std::abs(sample.depth - seed) > surface_tolerance(seed)) continue;
                    const Vec delta = sample.rgb - hole.rgb;
                    const double color_distance = dot(delta, delta);
                    if (color_distance > color_limit) continue;
                    const int dx = nx - x, dy = ny - y;
                    const double w = 1 / ((1 + dx * dx + dy * dy) * (1 + color_distance / color_limit));
                    const double iz = 1 / sample.depth;
                    mean = mean + Vec{double(dx), double(dy), iz} * w;
                    weight_sum += w;
                    xx += w * dx * dx;
                    yy += w * dy * dy;
                    zz += w * iz * iz;
                    xy += w * dx * dy;
                    xz += w * dx * iz;
                    yz += w * dy * iz;
                    ++support;
                }
            }
            if (support < 6) continue;
            mean = mean * (1 / weight_sum);
            xx = xx / weight_sum - mean.x * mean.x;
            yy = yy / weight_sum - mean.y * mean.y;
            zz = zz / weight_sum - mean.z * mean.z;
            xy = xy / weight_sum - mean.x * mean.y;
            xz = xz / weight_sum - mean.x * mean.z;
            yz = yz / weight_sum - mean.y * mean.z;
            const double det = xx * yy - xy * xy;
            if (det < 0.1) continue;  // Reject poorly constrained one-dimensional neighborhoods.
            const double a = (xz * yy - yz * xy) / det, b = (yz * xx - xz * xy) / det;
            const double residual = zz - a * xz - b * yz;
            const double inverse_depth_tolerance = 0.012 / (seed * seed);
            if (residual > inverse_depth_tolerance * inverse_depth_tolerance) continue;
            const double inverse_depth = mean.z - a * mean.x - b * mean.y;
            if (inverse_depth <= 0) continue;
            const double depth = 1 / inverse_depth;
            if (std::abs(depth - seed) <= surface_tolerance(seed)) completed[index].depth = depth;
        }
    }
    return completed;
}

void check_surfaces() {
    const Intrinsics k{3, 3, 1, 1, 0, 0};
    Frame image(k.cols * k.rows);
    rasterize({0, 0, 2, {255, 0, 0}}, {2, 0, 4, {0, 0, 255}}, {0, 2, 4, {0, 0, 255}}, k, image);
    if (std::abs(image[1].depth - 8.0 / 3.0) > 1e-10 || std::abs(image[1].rgb.x - 170) > 1e-10 ||
        std::abs(image[1].rgb.z - 85) > 1e-10 || valid(image[8]) ||
        !continuous(1.0, 1.01, 1.03) || continuous(1.0, 1.01, 1.3))
        throw std::runtime_error("Mesh perspective or depth-edge self-check failed");

    Frame plane(25, {{120, 110, 100}, 2});
    plane[12].depth = infinity;
    plane[0] = {{0, 0, 255}, infinity};
    const Frame fixed = complete_depth(plane, {5, 5, 1, 1, 0, 0}, 2);
    if (std::abs(fixed[12].depth - 2) > 1e-10 || valid(fixed[0]) || fixed[11].depth != plane[11].depth)
        throw std::runtime_error("Depth completion self-check failed");
}

void write_rgba(std::ostream& stream, const Frame& frame, std::vector<unsigned char>& bytes) {
    for (std::size_t i = 0; i < frame.size(); ++i) {
        const Vec rgb = frame[i].rgb;
        bytes[4 * i] = byte(rgb.x);
        bytes[4 * i + 1] = byte(rgb.y);
        bytes[4 * i + 2] = byte(rgb.z);
        bytes[4 * i + 3] = valid(frame[i]) ? 255 : 0;
    }
    if (!stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size()))
        throw std::runtime_error("Could not write rendered frame");
}

void run_paired(const std::filesystem::path& input, const std::filesystem::path& output) {
    check_surfaces();
    std::ifstream manifest(input / "pairs.txt");
    int source_cols = 0, source_rows = 0, target_cols = 0, target_rows = 0, count = 0;
    const auto valid_extent = [](int n) { return n >= 2 && n <= 4096; };
    if (!(manifest >> source_cols >> source_rows >> target_cols >> target_rows >> count) ||
        !valid_extent(source_cols) || !valid_extent(source_rows) ||
        !valid_extent(target_cols) || !valid_extent(target_rows) ||
        count < 1 || count > 10000) throw std::runtime_error("Invalid dimensions or frame count in pairs.txt");
    const double radius = std::max(double(target_cols) / source_cols, double(target_rows) / source_rows);
    const bool scaled = radius > 1;
    const double footprint = std::max(1.0, radius);
    const int repair_radius = std::max(2, int(std::ceil(5.0 / footprint)));

    std::filesystem::create_directories(output);
    std::ofstream rgba(output / "renders.rgba", std::ios::binary);
    std::ofstream timing(output / "timing.csv");
    if (!rgba || !timing) throw std::runtime_error("Could not open paired render outputs");
    timing << "frame,projection_ms,splat_ms,scaled_splat_ms,mesh_open_ms,mesh_gated_ms,repair_ms\n";
    timing << std::fixed << std::setprecision(4);
    std::vector<unsigned char> bytes(static_cast<std::size_t>(target_cols) * target_rows * 4);
    const auto read_k = [&](int cols, int rows) {
        Intrinsics k{cols, rows};
        if (!(manifest >> k.fx >> k.fy >> k.u0 >> k.v0) || k.fx <= 0 || k.fy <= 0)
            throw std::runtime_error("Invalid calibration in pairs.txt");
        return k;
    };
    const auto read_pose = [&]() {
        double r00 = 0;
        if (!(manifest >> r00)) throw std::runtime_error("Missing pose in pairs.txt");
        return read_camera(manifest, r00);
    };
    using Clock = std::chrono::steady_clock;
    const auto elapsed_ms = [](Clock::time_point begin, Clock::time_point end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };

    for (int i = 0; i < count; ++i) {
        const Intrinsics source_k = read_k(source_cols, source_rows);
        const Intrinsics target_k = read_k(target_cols, target_rows);
        const Camera source_camera = read_pose(), target_camera = read_pose();
        const Frame source = read_rgbd(input / (std::to_string(i) + ".rgbd"), source_k);
        const auto begin = Clock::now();
        std::vector<Sample> grid(source.size()), samples;
        samples.reserve(source.size());
        for (int y = 0; y < source_rows; ++y) {
            for (int x = 0; x < source_cols; ++x) {
                const int index = y * source_cols + x;
                const Sample s = project_pixel(source[index], source_k.ray(x, y), source_camera, target_camera, target_k);
                grid[index] = s;
                if (std::isfinite(s.z) && s.x > -footprint && s.y > -footprint &&
                    s.x < target_cols - 1 + footprint && s.y < target_rows - 1 + footprint)
                    samples.push_back(s);
            }
        }
        const auto projected = Clock::now();
        const Frame splatted = bilinear(samples, depth_gate, target_k);
        const auto splat_done = Clock::now();
        write_rgba(rgba, splatted, bytes);
        timing << i << ',' << elapsed_ms(begin, projected) << ',' << elapsed_ms(projected, splat_done);

        Frame wider;
        double scaled_ms = 0;
        if (scaled) {
            const auto start = Clock::now();
            wider = bilinear(samples, depth_gate, target_k, radius);
            scaled_ms = elapsed_ms(start, Clock::now());
            write_rgba(rgba, wider, bytes);
        }
        timing << ',' << scaled_ms;
        const Frame& fallback = scaled ? wider : splatted;
        Frame gated;
        for (bool gate : {false, true}) {
            const auto start = Clock::now();
            Frame mesh = mesh_render(source, grid, source_k, target_k, gate);
            // A splat covers isolated valid points and rejected or subpixel triangles.
            for (std::size_t j = 0; j < mesh.size(); ++j)
                if (!valid(mesh[j])) mesh[j] = fallback[j];
            timing << ',' << elapsed_ms(start, Clock::now());
            write_rgba(rgba, mesh, bytes);
            if (gate) gated = std::move(mesh);
        }

        const auto repair_start = Clock::now();
        const Frame completed = complete_depth(source, source_k, repair_radius);
        std::vector<Sample> repaired_samples;
        for (int y = 0; y < source_rows; ++y) {
            for (int x = 0; x < source_cols; ++x) {
                const int j = y * source_cols + x;
                if (valid(source[j]) || !valid(completed[j])) continue;
                const Sample s = project_pixel(completed[j], source_k.ray(x, y), source_camera, target_camera, target_k);
                grid[j] = s;
                if (std::isfinite(s.z) && s.x > -footprint && s.y > -footprint &&
                    s.x < target_cols - 1 + footprint && s.y < target_rows - 1 + footprint)
                    repaired_samples.push_back(s);
            }
        }
        const Frame repaired_mesh = mesh_render(completed, grid, source_k, target_k, true);
        const Frame repaired_splat = bilinear(repaired_samples, depth_gate, target_k, footprint);
        // Preserve existing projected colors. Only infer geometry at missing source depth.
        for (std::size_t j = 0; j < gated.size(); ++j) {
            if (!valid(gated[j]))
                gated[j] = valid(repaired_mesh[j]) ? repaired_mesh[j] : repaired_splat[j];
        }
        timing << ',' << elapsed_ms(repair_start, Clock::now()) << '\n';
        write_rgba(rgba, gated, bytes);
    }
    std::string extra;
    if (manifest >> extra) throw std::runtime_error("Unexpected records at end of pairs.txt");
    timing.flush();
    rgba.flush();
    if (!timing || !rgba) throw std::runtime_error("Could not finish writing paired outputs");
}

double squared_distance(Vec a, Vec b) { const Vec d = a - b; return dot(d, d); }

void report_multiview(std::ostream& csv, std::size_t target, std::size_t primary, const char* method,
                      const Frame& predicted, const Frame& reference, const Frame& first,
                      const Frame& second) {
    constexpr double depth_tolerance = 0.05;
    std::size_t valid_ref = 0, missing = 0, agree = 0, shared = 0, added = 0, added_agree = 0;
    double shared_error = 0, added_error = 0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        if (!valid(reference[i])) continue;
        ++valid_ref;
        if (!valid(predicted[i])) {
            ++missing;
            continue;
        }
        const bool depth_agrees = std::abs(predicted[i].depth - reference[i].depth) <= depth_tolerance;
        agree += depth_agrees;
        const double error = squared_distance(rounded(predicted[i].rgb), reference[i].rgb);
        if (valid(first[i]) && valid(second[i])) {
            ++shared;
            shared_error += error;
        }
        if (!valid(first[i])) {
            ++added;
            added_agree += depth_agrees;
            added_error += error;
        }
    }
    if (valid_ref == 0) throw std::runtime_error("No valid reference depth");
    csv << target << ',' << primary << ',' << method << ',' << valid_ref << ','
        << 100.0 * missing / valid_ref << ',' << 100.0 * agree / valid_ref << ','
        << psnr(shared_error, shared) << ',' << added << ',' << added_agree << ','
        << psnr(added_error, added) << '\n';
}

void run_multiview(const std::filesystem::path& input, const std::filesystem::path& output) {
    std::ifstream manifest(input / "cameras.txt");
    Intrinsics k;
    if (!(manifest >> k.cols >> k.rows >> k.fx >> k.fy >> k.u0 >> k.v0) ||
        k.cols <= 0 || k.cols > 8192 || k.rows <= 0 || k.rows > 8192 || k.fx <= 0 || k.fy <= 0)
        throw std::runtime_error("Invalid intrinsics in cameras.txt");
    std::vector<Camera> cameras;
    for (double r00; manifest >> r00;) {
        cameras.push_back(read_camera(manifest, r00));
    }
    if (!manifest.eof() || cameras.size() < 3) throw std::runtime_error("Expected at least three camera poses");

    std::filesystem::create_directories(output);
    std::ofstream csv(output / "multiview.csv");
    if (!csv) throw std::runtime_error("Could not open multiview.csv");
    csv << "target,primary,method,reference_pixels,hole_pct,depth_agree_5cm_pct,shared_psnr_db,"
           "added_pixels,added_depth_agree_5cm_pixels,added_psnr_db\n";
    csv << std::fixed << std::setprecision(6);

    const auto load_frame = [&](std::size_t i) { return read_rgbd(input / (std::to_string(i) + ".rgbd"), k); };
    Frame left = load_frame(0), reference = load_frame(1);
    for (std::size_t target = 1; target + 1 < cameras.size(); ++target) {
        Frame right = load_frame(target + 1);
        const Camera& view = cameras[target];
        const bool left_is_nearer = squared_distance(cameras[target - 1].center, view.center) <=
                                    squared_distance(cameras[target + 1].center, view.center);
        const std::size_t primary = left_is_nearer ? target - 1 : target + 1;
        const std::size_t auxiliary = left_is_nearer ? target + 1 : target - 1;
        auto primary_samples = reproject(left_is_nearer ? left : right, cameras[primary], view, k);
        const auto auxiliary_samples = reproject(left_is_nearer ? right : left, cameras[auxiliary], view, k);
        const Frame single = bilinear(primary_samples, depth_gate, k);
        const Frame second = bilinear(auxiliary_samples, depth_gate, k);

        // Reuse the existing depth buffer for global nearest-surface selection.
        primary_samples.insert(primary_samples.end(), auxiliary_samples.begin(), auxiliary_samples.end());
        const Frame joint = bilinear(primary_samples, depth_gate, k);

        // Fill holes with the auxiliary view while preserving the preferred view's existing pixels.
        Frame filled = single;
        for (std::size_t i = 0; i < filled.size(); ++i)
            if (!valid(filled[i])) filled[i] = second[i];

        const std::array<std::pair<const char*, const Frame*>, 4> methods{{
            {"near_single", &single}, {"far_single", &second}, {"joint_splat", &joint}, {"fill_only", &filled}
        }};
        for (const auto& [name, frame] : methods)
            report_multiview(csv, target, primary, name, *frame, reference, single, second);
        if (target == cameras.size() / 2) {
            write_frame(output / "target.tga", reference, k);
            for (const auto& [name, frame] : methods)
                write_frame(output / (std::string(name) + ".tga"), *frame, k);
        }
        left = std::move(reference);
        reference = std::move(right);
    }
    if (!csv) throw std::runtime_error("Could not write multiview.csv");
    std::cout << "Compared four methods on " << cameras.size() - 2 << " held-out frames.\n";
}

void check_occlusion() {
    const std::vector<Sample> probes{{15.25, 12.25, 2, {255, 0, 0}}, {15.25, 12.25, 4, {0, 0, 255}}};
    const auto index = 12 * width + 15;
    const Pixel gated = bilinear(probes, depth_gate)[index];
    const Pixel blended = bilinear(probes, infinity)[index];
    if (std::abs(gated.rgb.x - 255) > 1e-9 || gated.rgb.z != 0 ||
        std::abs(blended.rgb.x - 127.5) > 1e-9 || std::abs(blended.rgb.z - 127.5) > 1e-9)
        throw std::runtime_error("Occlusion self-check failed");
}
}  // namespace

int main(int argc, char** argv) try {
    if (argc >= 2 && std::string(argv[1]) == "--paired") {
        if (argc != 4) throw std::runtime_error("Usage: rgbd_demo --paired input output");
        run_paired(argv[2], argv[3]);
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--multiview") {
        if (argc != 4) throw std::runtime_error("Usage: rgbd_demo --multiview input output");
        run_multiview(argv[2], argv[3]);
        return 0;
    }
    if (argc > 2)
        throw std::runtime_error("Usage: rgbd_demo [output_directory] or rgbd_demo --multiview/--paired input output");
    const std::filesystem::path output = argc == 2 ? argv[1] : "results";
    std::filesystem::create_directories(output);
    std::ofstream csv(output / "metrics.csv");
    if (!csv) throw std::runtime_error("Could not open metrics.csv");
    csv << "scene,shift,yaw_deg,method,hole_pct,full_psnr_db,common_valid_psnr_db\n";
    csv << std::fixed << std::setprecision(6);

    const std::array<Scene, 2> scenes{{
        {{{{-0.58, -0.02, 2.8}, 0.73, {239, 117, 79}}, {{0.75, 0.14, 3.85}, 0.97, {62, 166, 184}}, {{0.22, -0.86, 4.6}, 0.63, {238, 198, 84}}}},
        {{{{0.37, 0.30, 2.55}, 0.67, {83, 157, 219}}, {{-0.82, -0.12, 4.15}, 1.13, {218, 126, 176}}, {{0.93, -0.62, 4.5}, 0.72, {94, 185, 143}}}}
    }};
    const std::array<std::array<double, 2>, 4> poses{{{0, 0}, {0.08, 1.5}, {-0.14, -2}, {0.22, 3.0}}};
    const Camera source = Camera::planar(0, 0);
    check_occlusion();
    for (std::size_t scene = 0; scene < scenes.size(); ++scene) {
        const Frame source_rgbd = render(source, scenes[scene]);
        for (std::size_t pose = 0; pose < poses.size(); ++pose) {
            const auto [shift, yaw] = poses[pose];
            const Camera target = Camera::planar(shift, yaw);
            const auto samples = reproject(source_rgbd, source, target);
            const Frame reference = render(target, scenes[scene]);
            const Frame z_buffer = nearest(samples);
            const Frame blended = bilinear(samples, infinity);
            const Frame gated = bilinear(samples, depth_gate);
            const std::array<std::pair<const char*, const Frame*>, 3> methods{{
                {"nearest_z", &z_buffer}, {"bilinear", &blended}, {"depth_gated", &gated}
            }};
            for (const auto& [name, frame] : methods) {
                const auto m = evaluate(*frame, reference, z_buffer, gated);
                if (pose == 0 && (m.holes != 0 || !std::isinf(m.all_psnr)))
                    throw std::runtime_error("Identity reprojection self-check failed");
                csv << scene + 1 << ',' << shift << ',' << yaw << ',' << name << ','
                    << m.holes << ',' << m.all_psnr << ',' << m.common_psnr << '\n';
            }
            if (scene == 0 && pose == 2) {
                write_frame(output / "source.tga", source_rgbd);
                write_frame(output / "reference.tga", reference);
                for (const auto& [name, frame] : methods)
                    write_frame(output / (std::string(name) + ".tga"), *frame);
            }
        }
    }
    if (!csv) throw std::runtime_error("Could not finish writing metrics.csv");
    std::cout << "Passed identity and occlusion checks. Wrote 24 evaluation rows to " << output << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
