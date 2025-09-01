const std = @import("std");
const parser = @import("parser.zig");

const builtin = @import("builtin");

extern fn jsPrint(ptr: [*]const u8, len: usize) void;
extern fn jsFlush() void;
const Logger = struct {
    pub const Error = error{};
    pub const Writer = std.io.Writer(void, Error, write);
    fn write(_: void, bytes: []const u8) Error!usize {
        jsPrint(bytes.ptr, bytes.len);
        return bytes.len;
    }
};

const logger = Logger.Writer{ .context = {} };
pub fn print(comptime format: []const u8, args: anytype) void {
    logger.print(format, args) catch return;
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
    const parseResult = myParser.parse() catch |err| {
        print("couldn't parse: {}", .{err});
        return;
    };
    defer parseResult.arena.deinit();
    print("Input \"{s}\" output \"{s}\"", .{ val, parseResult.expression });
}
