const std = @import("std");
const parser = @import("parser.zig");

const builtin = @import("builtin");

extern fn jsPrint(ptr: [*]const u8, len: usize) void;
extern fn jsFlush() void;
const Logger = struct {
    pub const Error = error{};
    writer: std.io.Writer,
    fn init() @This() {
        return .{
            .writer = .{
                .buffer = &[_]u8{},
                .vtable = &.{ .drain = @This().drain },
            },
        };
    }
    fn drain(_: *std.io.Writer, data: []const []const u8, splat: usize) Error!usize {
        var written: usize = 0;
        for (data[0 .. data.len - 1]) |slice| {
            jsPrint(slice.ptr, slice.len);
            written += slice.len;
        }
        for (0..splat) |_| jsPrint(data[data.len - 1].ptr, data[data.len - 1].len);
        return written + splat * data[data.len - 1].len;
    }
};

pub fn print(comptime format: []const u8, args: anytype) void {
    var logger = Logger.init();
    logger.writer.print(format, args) catch return;
    jsFlush();
}

fn from_ptr_len(ptr: usize, len: usize) []u8 {
    const ptrValue: [*]u8 = @ptrFromInt(ptr);
    return ptrValue[0..len];
}

export fn alloc(size: usize) usize {
    const value = std.heap.page_allocator.alloc(u8, size) catch {
        return 0;
    };
    return @intFromPtr(value.ptr);
}

export fn free(ptr: usize, len: usize) void {
    std.heap.page_allocator.free(from_ptr_len(ptr, len));
}

export fn eval(ptr: usize, len: usize) void {
    const val = from_ptr_len(ptr, len);
    var myParser = parser.Parser.new(std.heap.page_allocator, val) catch |err| {
        print("couldn't create parser: {}", .{err});
        return;
    };
    var parseResult = myParser.parse() catch |err| {
        print("couldn't parse: {}", .{err});
        return;
    };
    defer parseResult.arena.deinit();

    var logger = Logger.init();
    logger.writer.print("Input <math display=\"inline\">", .{}) catch return;
    parseResult.expression.formatMathML(&logger.writer, parseResult.arena.allocator(), @enumFromInt(0)) catch return;
    logger.writer.print("</math>", .{}) catch return;

    parseResult.simplify() catch |err| {
        print("couldn't simplify: {}", .{err});
        return;
    };
    logger.writer.print("Output <math display=\"inline\">", .{}) catch return;
    parseResult.expression.formatMathML(&logger.writer, parseResult.arena.allocator(), @enumFromInt(0)) catch return;
    logger.writer.print("</math>", .{}) catch return;
    jsFlush();
}
