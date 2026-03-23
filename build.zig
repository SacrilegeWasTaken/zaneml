const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("zaneml", .{
        .root_source_file = b.path("lib/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    // Allow @cImport("bridge.h") in metal/engine.zig to find the header
    mod.addIncludePath(b.path("lib/metal"));

    const example_imports: []const std.Build.Module.Import = &.{
        .{ .name = "zaneml", .module = mod },
    };

    // ── Examples ─────────────────────────────────────────────────────────────

    const main_exe = addExample(b, "main", "examples/main.zig", target, optimize, example_imports);
    const perceptron_exe = addExample(b, "perceptron", "examples/perceptron.zig", target, optimize, example_imports);
    const transformer_exe = addExample(b, "transformer", "examples/transformer.zig", target, optimize, example_imports);
    const autograd_exe = addExample(b, "autograd", "examples/autograd.zig", target, optimize, example_imports);

    b.installArtifact(main_exe);

    addRunStep(b, main_exe, "run", "Run examples/main.zig");
    addRunStep(b, main_exe, "main", "Run examples/main.zig");
    addRunStep(b, perceptron_exe, "perceptron", "Run examples/perceptron.zig");
    addRunStep(b, transformer_exe, "transformer", "Run examples/transformer.zig");
    addRunStep(b, autograd_exe, "autograd", "Run examples/autograd.zig (XOR via Tape API)");

    // ── Tests ────────────────────────────────────────────────────────────────

    // Unit tests — discovered transitively from lib/root.zig
    const unit_tests = b.addTest(.{ .root_module = mod });
    const run_unit_tests = b.addRunArtifact(unit_tests);

    // Integration tests — tests/ directory, imports zaneml as a dependency
    const integration_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/integration.zig"),
            .target = target,
            .optimize = optimize,
            .imports = example_imports,
        }),
    });
    const run_integration_tests = b.addRunArtifact(integration_tests);

    const test_step = b.step("test", "Run all tests (unit + integration)");
    test_step.dependOn(&run_unit_tests.step);
    test_step.dependOn(&run_integration_tests.step);

    const unit_test_step = b.step("test-unit", "Run unit tests only (inline in lib/)");
    unit_test_step.dependOn(&run_unit_tests.step);

    const integration_test_step = b.step("test-integration", "Run integration tests only (tests/)");
    integration_test_step.dependOn(&run_integration_tests.step);

    // Metal smoke tests
    const metal_test_exe = addExample(b, "metal-test", "tests/metal_test.zig", target, optimize, example_imports);
    addRunStep(b, metal_test_exe, "test-metal", "Run Metal compute smoke tests");
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Create an executable with zaneml imports + Metal linking (on macOS).
fn addExample(
    b: *std.Build,
    name: []const u8,
    path: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    imports: []const std.Build.Module.Import,
) *std.Build.Step.Compile {
    const exe = b.addExecutable(.{
        .name = name,
        .root_module = b.createModule(.{
            .root_source_file = b.path(path),
            .target = target,
            .optimize = optimize,
            .imports = imports,
        }),
    });
    // Link Metal bridge + frameworks so any backend (.cpu or .metal) works
    exe.addIncludePath(b.path("lib/metal"));
    exe.addCSourceFile(.{
        .file = b.path("lib/metal/bridge.m"),
        .flags = &.{"-fobjc-arc"},
    });
    exe.linkFramework("Metal");
    exe.linkFramework("Foundation");
    return exe;
}

fn addRunStep(b: *std.Build, exe: *std.Build.Step.Compile, name: []const u8, desc: []const u8) void {
    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| run_cmd.addArgs(args);
    const step = b.step(name, desc);
    step.dependOn(&run_cmd.step);
}
