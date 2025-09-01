const std = @import("std");
const expect = std.testing.expectEqualDeep;
const FormatOptions = std.fmt.FormatOptions;
const root = @import("root");
pub const print: fn (
    comptime format: []const u8,
    args: anytype,
) void = if (@hasDecl(root, "print")) root.print else std.debug.print;

/// Splits the string into individual pieces that are easy to manipulate
/// e.g. 'sin(2x)' -> ['sin', '(', '2', 'x', ')']
pub const Tokeniser = struct {
    /// Possible types for each token
    const TokenType = enum { Integer, Literal, Add, Subtract, Multiply, Divide, Power, OpenBracket, CloseBracket };

    /// Token including type, position, and length. This position information can be used to determine what integer literal is used.
    const Token = struct {
        ty: TokenType,
        index: usize,
        length: usize,
    };

    /// The text that the user input
    source: []const u8,
    /// The current index that the tokeniser is looking at
    index: usize,
    /// The index that the current token started at
    start: usize,

    const Self = @This();

    pub fn new(source: []const u8) Self {
        return Self{ .source = source, .index = 0, .start = 0 };
    }

    /// Create a token starting at the Tokeniser's start and ending at the tokeniser's current index.
    fn makeToken(self: Self, ty: TokenType) Token {
        return Token{ .ty = ty, .index = self.start, .length = self.index - self.start };
    }

    /// You are done when you have looked at all of the input text
    pub fn isDone(self: Self) bool {
        return self.index >= self.source.len;
    }

    pub fn skipWhitespace(self: *Self) void {
        while (!self.isDone() and (self.peek() == ' ' or self.peek() == '\r' or self.peek() == '\n')) {
            self.index += 1;
        }
    }

    /// Move to the next character, returning the one that used to be at.
    fn advance(self: *Self) u8 {
        const c = self.source[self.index];
        self.index += 1;
        return c;
    }

    /// Look at the current character without advancing
    fn peek(self: Self) u8 {
        return self.source[self.index];
    }

    /// Get the next token (if possible)
    pub fn next(self: *Self) !?Token {
        self.skipWhitespace();
        if (self.isDone()) {
            return null;
        }
        self.start = self.index;
        switch (self.advance()) {
            '(' => return self.makeToken(TokenType.OpenBracket),
            ')' => return self.makeToken(TokenType.CloseBracket),
            '*' => return self.makeToken(TokenType.Multiply),
            '/' => return self.makeToken(TokenType.Divide),
            '+' => return self.makeToken(TokenType.Add),
            '-' => return self.makeToken(TokenType.Subtract),
            '^' => return self.makeToken(TokenType.Power),
            '0'...'9' => {
                while (!self.isDone() and '0' <= self.peek() and self.peek() <= '9') {
                    self.index += 1;
                }
                return self.makeToken(TokenType.Integer);
            },
            'a'...'z', 'A'...'Z', '_' => {
                while (!self.isDone() and (('a' <= self.peek() and self.peek() <= 'z') or ('A' <= self.peek() and self.peek() <= 'Z') or self.peek() == '_')) {
                    self.index += 1;
                }
                return self.makeToken(TokenType.Literal);
            },
            else => {
                print("whilst tokenising, found invalid character '{}' at index {} in source \"{s}\"\n", .{ self.source[self.index - 1], self.index - 1, self.source });
                return error.UnexpectedChar;
            },
        }
    }
};

// Test tokeniser
test "Tokenise Int" {
    var tokeniser = Tokeniser.new(" \n\r\n 93   ");
    try expect((try tokeniser.next()).?.ty, Tokeniser.TokenType.Integer);
    try expect((try tokeniser.next()), null);
}
test "Tokenise Literal" {
    var tokeniser = Tokeniser.new("  aA_BzZ  ");
    try expect((try tokeniser.next()).?.ty, Tokeniser.TokenType.Literal);
    try expect((try tokeniser.next()), null);
}
test "Tokenise IntLit" {
    var tokeniser = Tokeniser.new("2a");
    try expect((try tokeniser.next()).?.ty, Tokeniser.TokenType.Integer);
    try expect((try tokeniser.next()).?.ty, Tokeniser.TokenType.Literal);
    try expect((try tokeniser.next()), null);
}

/// An expression is the equivilant of an abstract base class.
const Expression = union(enum) {
    integer: i32,
    variable: []const u8,
    product: std.ArrayListUnmanaged(Expression),
    sum: std.ArrayListUnmanaged(Expression),
    power: *[2]Expression,

    const Self = @This();
    /// Useful for displaying expressions nicely
    pub fn format(self: *const Self, writer: anytype) !void {
        switch (self.*) {
            .integer => |value| try writer.print("{}", .{value}),
            .variable => |name| try writer.print("{s}", .{name}),
            .product => |terms| {
                for (terms.items) |value| {
                    try writer.print("({f})", .{value});
                }
            },
            .sum => |terms| {
                for (terms.items, 0..terms.items.len) |value, index| {
                    if (index != 0) {
                        try writer.print(" + ", .{});
                    }
                    try value.format(writer);
                }
            },
            .power => |power| try writer.print("( {f} )^( {f} )", .{ power[0], power[1] }),
        }
    }
    /// Construct a power expression
    pub fn newPower(base: Expression, exponent: Expression, alloc: std.mem.Allocator) std.mem.Allocator.Error!Expression {
        const expressions = try alloc.create([2]Expression);
        expressions[0] = base;
        expressions[1] = exponent;
        return Expression{ .power = expressions };
    }
    /// Construct a sum expression
    pub fn newSum(slice: []const Expression, alloc: std.mem.Allocator) std.mem.Allocator.Error!Expression {
        var list: std.ArrayListUnmanaged(Expression) = .empty;
        try list.appendSlice(alloc, slice);
        return Expression{ .sum = list };
    }
    /// Construct a product expression
    pub fn newProduct(slice: []const Expression, alloc: std.mem.Allocator) std.mem.Allocator.Error!Expression {
        var list: std.ArrayListUnmanaged(Expression) = .empty;
        try list.appendSlice(alloc, slice);
        return Expression{ .product = list };
    }
    /// Construct an integer expression
    pub fn newInteger(value: i32) Expression {
        return Expression{ .integer = value };
    }
    /// Construct a variable epxression
    pub fn newVariable(name: []const u8) Expression {
        return Expression{ .variable = name };
    }
    /// Negates the current expression by multiplying by -1
    fn negate(self: Self, alloc: std.mem.Allocator) std.mem.Allocator.Error!Expression {
        return Self.newProduct(&[_]Expression{ self, Expression.newInteger(-1) }, alloc);
    }
    /// Reciprical of the current expression by rasining to the power of -1
    fn recip(self: Self, alloc: std.mem.Allocator) std.mem.Allocator.Error!Expression {
        return Self.newPower(self, Expression.newInteger(-1), alloc);
    }
    // fn cmp(_: void, a: Self, b: Self) bool {
    //     switch (a) {
    //         .integer => return false,
    //         .variable => |name| try writer.print("{s}", .{name}),
    //         .product => |terms| {
    //             for (terms.items) |value| {
    //                 try writer.print("({})", .{value});
    //             }
    //         },
    //         .sum => |terms| {
    //             for (terms.items, 0..terms.items.len) |value, index| {
    //                 if (index != 0) {
    //                     try writer.print(" + ", .{});
    //                 }
    //                 try value.format(fmt, options, writer);
    //             }
    //         },
    //         .power => |power| try writer.print("( {} )^( {} )", .{ power[0], power[1] }),
    //     }
    //     return a > b;
    // }
    /// Collapse e.g. a sum of sums becomes a single sum
    fn collapsed_nested(self: *Self, alloc: std.mem.Allocator) std.mem.Allocator.Error!void {
        switch (self.*) {
            .product => |*terms| {
                var index: usize = 0;
                while (index < terms.items.len) {
                    try terms.items[index].collapsed_nested(alloc);
                    switch (terms.items[index]) {
                        .product => |*product| try terms.replaceRange(alloc, index, 1, product.items),
                        else => index += 1,
                    }
                }
                // std.mem.sort(u8, terms, {}, comptime std.sort.desc(u8));
                if (terms.items.len == 0) {
                    self.* = Expression.newInteger(1);
                }
            },
            .sum => |*terms| {
                var index: usize = 0;
                while (index < terms.items.len) {
                    try terms.items[index].collapsed_nested(alloc);
                    switch (terms.items[index]) {
                        .sum => |*sum| try terms.replaceRange(alloc, index, 1, sum.items),
                        else => index += 1,
                    }
                }
                // std.mem.sort(u8, terms, {}, comptime std.sort.desc(u8));
                if (terms.items.len == 0) {
                    self.* = Expression.newInteger(0);
                }
            },
            .power => |power| {
                try power[0].collapsed_nested(alloc);
                try power[1].collapsed_nested(alloc);
                switch (power[0]) {
                    .power => |inner| {
                        power[0] = inner[0];
                        power[1] = try Expression.newProduct(&[_]Expression{ inner[1], power[1] }, alloc);
                        try power[1].collapsed_nested(alloc);
                    },
                    else => {},
                }
            },

            .integer, .variable => {},
        }
    }
    fn without_multiple(self: *Self, alloc: std.mem.Allocator) std.mem.Allocator.Error!i32 {
        switch (self.*) {
            .product => |*p| {
                var count: i32 = 1;
                var index: usize = 0;
                while (index < p.items.len) {
                    switch (p.items[index]) {
                        .integer => |i| {
                            count *= i;
                            _ = p.swapRemove(index);
                        },
                        else => index += 1,
                    }
                }
                try self.collapsed_nested(alloc);
                return count;
            },
            .integer => |count| {
                self.* = Expression.newInteger(1);
                return count;
            },
            else => return 1,
        }
    }
    // fn consolidate_sum(self: *self, other: Expression) std.mem.Allocator.Error!bool {
    //     switch (other) {
    //         .product =>
    //     }
    // }
    /// Collect like terms e.g. x + 2 + 2x becomes 3x + 2
    fn collect_like_terms(self: *Self, alloc: std.mem.Allocator) std.mem.Allocator.Error!void {
        switch (self.*) {
            .sum => |*oldTerms| {
                var newTerms: std.ArrayListUnmanaged(Expression) = .empty;
                for (oldTerms.items) |*oldTerm| {
                    const oldMultiple = try oldTerm.without_multiple(alloc);
                    var canConsolidate = false;
                    for (newTerms.items) |*newTerm| {
                        var new = try newTerm.without_multiple(alloc);
                        canConsolidate = newTerm == oldTerm;
                        print("New {f} old {f} consolidate {}\n", .{ newTerm, oldTerm, canConsolidate });
                        if (canConsolidate) {
                            new += oldMultiple;
                        }
                        newTerm.* = try Expression.newProduct(&[_]Expression{ Expression.newInteger(new), newTerm.* }, alloc);
                        try newTerm.collapsed_nested(alloc);
                        if (canConsolidate) {
                            break;
                        }
                    }
                    if (!canConsolidate) {
                        var value = try Expression.newProduct(&[_]Expression{ Expression.newInteger(oldMultiple), oldTerm.* }, alloc);
                        try value.collapsed_nested(alloc);
                        try newTerms.append(alloc, value);
                    }
                }
                oldTerms.* = newTerms;
            },
            else => {},
        }
        try self.collapsed_nested(alloc);
    }
};

/// The parser converts the input text into an abstract synatx tree (AST)
pub const Parser = struct {
    arena: std.heap.ArenaAllocator,

    tokeniser: Tokeniser,
    previous: ?Token,
    current: ?Token,

    const Prec = enum { Sum, Product, Exponent, Prefix, Call };
    const Self = @This();
    const TokenType = Tokeniser.TokenType;
    const Token = Tokeniser.Token;

    pub fn new(alloc: std.mem.Allocator, source: []const u8) !Parser {
        var tokeniser = Tokeniser.new(source);
        const previous = try tokeniser.next();
        return Parser{ .arena = std.heap.ArenaAllocator.init(alloc), .tokeniser = tokeniser, .previous = null, .current = previous };
    }

    /// Helper function to access the alloc
    fn allocator(self: *Self) std.mem.Allocator {
        return self.arena.allocator();
    }

    /// Move to the next token
    fn advance(self: *Self) !?Token {
        self.previous = self.current;
        self.current = try self.tokeniser.next();
        return self.previous;
    }

    /// Produce an expression based on a group (thing in brackets)
    fn group(self: *Self) !Expression {
        // Parse with lowest prec
        const expression = try self.parseWithPrec(@enumFromInt(0));
        if (self.current.?.ty != TokenType.CloseBracket) {
            print("Expected ')', found {?}\n", .{self.current});
            return error.NoCloseBracket;
        }
        _ = try self.advance();
        return expression;
    }

    /// Produce an expression based on a number literal
    fn number(self: *Self) !Expression {
        const token = self.previous.?;
        // There is probably a way to convert a slice to a number but I implemented it manually for fun
        var value: i32 = 0;
        for (token.index..token.index + token.length) |index| {
            value *= 10;
            value += self.tokeniser.source[index] - '0';
        }
        return Expression.newInteger(value);
    }

    /// Produce an expression based on a variable literal
    fn variable(self: *Self) !Expression {
        const token = self.previous.?;
        return Expression.newVariable(self.tokeniser.source[token.index .. token.index + token.length]);
    }

    /// The unary negation.
    fn negate(self: *Self) !Expression {
        const inner = try self.parseWithPrec(Prec.Prefix);
        return inner.negate(self.allocator());
    }

    /// Binary epxressions
    fn binary(self: *Self, token: TokenType, prec: Prec, lhs: Expression) !Expression {
        // Parse with the next prec up
        var rhs = try self.parseWithPrec(@enumFromInt(@intFromEnum(prec) + 1));
        const alloc = self.allocator();
        return switch (token) {
            TokenType.Add => Expression.newSum(&[_]Expression{ lhs, rhs }, alloc),
            TokenType.Subtract => Expression.newSum(&[_]Expression{ lhs, try rhs.negate(alloc) }, alloc),
            TokenType.Multiply => Expression.newProduct(&[_]Expression{ lhs, rhs }, alloc),
            TokenType.Divide => Expression.newProduct(&[_]Expression{ lhs, try rhs.recip(alloc) }, alloc),
            TokenType.Power => Expression.newPower(lhs, rhs, alloc),
            else => {
                print("whilst parsing binary expression, encountered invalid token type {}\n", .{token});
                return error.UnexpectedToken;
            },
        };
    }

    /// Represents the pratt parsing functions for a particular token type
    const Rule = struct { prefix: ?*const fn (*Parser) anyerror!Expression, infix: ?*const fn (*Parser, TokenType, Prec, Expression) anyerror!Expression, prec: Prec };
    /// Retreive the pratt parsing functions for the token type
    pub fn rule_for_token(token_type: TokenType) ?Rule {
        return switch (token_type) {
            TokenType.OpenBracket => Rule{ .prefix = group, .infix = null, .prec = Prec.Prefix },
            TokenType.Integer => Rule{ .prefix = number, .infix = null, .prec = Prec.Prefix },
            TokenType.Literal => Rule{ .prefix = variable, .infix = null, .prec = Prec.Prefix },
            TokenType.Subtract => Rule{ .prefix = negate, .infix = binary, .prec = Prec.Sum },
            TokenType.Add => Rule{ .prefix = null, .infix = binary, .prec = Prec.Sum },
            TokenType.Multiply => Rule{ .prefix = null, .infix = binary, .prec = Prec.Product },
            TokenType.Divide => Rule{ .prefix = null, .infix = binary, .prec = Prec.Product },
            TokenType.Power => Rule{ .prefix = null, .infix = binary, .prec = Prec.Exponent },
            else => null,
        };
    }

    /// Gobbles all unary and then infix with >=targetPrec
    pub fn parseWithPrec(self: *Self, targetPrec: Prec) anyerror!Expression {
        const prefixToken = try self.advance() orelse return error.NoToken;

        // Get the prefix expression (will be the lhs of the expression)
        var prefix: ?*const fn (*Parser) anyerror!Expression = null;
        if (Self.rule_for_token(prefixToken.ty)) |rule| {
            prefix = rule.prefix;
        }
        const prefixFn = prefix orelse {
            print("could not find rule for token {} in a prefix postion\n", .{prefixToken});
            return error.UnexpectedToken;
        };

        var expression = try prefixFn(self);

        // While there are tokens left, keep attempting to get infix expressions
        while (self.current) |current| {
            const rule = Self.rule_for_token(current.ty) orelse break;
            const infix = rule.infix orelse {
                // Could not find rule for token in an infix postion, assuming implict multiplication
                // If the prec of the rule is lower than our target, stop gobbling.
                if (@intFromEnum(Prec.Product) < @intFromEnum(targetPrec)) {
                    break;
                }
                expression = try binary(self, TokenType.Multiply, Prec.Product, expression);

                continue;
            };

            // If the prec of the rule is lower than our target, stop gobbling.
            if (@intFromEnum(rule.prec) < @intFromEnum(targetPrec)) {
                break;
            }
            // Advance over the token that made us infix
            _ = try self.advance();
            expression = try infix(self, current.ty, rule.prec, expression);
        }
        return expression;
    }

    pub const ParseResult = struct { expression: Expression, arena: std.heap.ArenaAllocator };
    pub fn parse(self: *Self) !ParseResult {
        // Parse with lowest prec
        var expression = try self.parseWithPrec(@enumFromInt(0));
        if (self.current != null) {
            print("Unparsed token {?}\n", .{self.current});
        }
        try expression.collapsed_nested(self.allocator());
        return .{ .expression = expression, .arena = self.arena };
    }
    pub fn parse_and_simplify(self: *Self) !ParseResult {
        var result = try self.parse();
        try result.expression.collect_like_terms(self.allocator());
        return result;
    }
};

/// A utilty for testing the parser
fn testParser(source: []const u8) !Parser.ParseResult {
    var parser = try Parser.new(std.testing.allocator, source);
    const parsed = try parser.parse();
    print("source {s}\nparsed {f}\n", .{ source, parsed.expression });
    return parsed;
}

test "Parse IntLit" {
    const parsed = try testParser("42");
    defer parsed.arena.deinit();
    try expect(Expression.newInteger(42), parsed.expression);
}

test "Parse Double negative" {
    var parsed = try testParser("--2");
    defer parsed.arena.deinit();
    const expected = &[_]Expression{ Expression.newInteger(2), Expression.newInteger(-1), Expression.newInteger(-1) };
    try expect(expected, parsed.expression.product.items);
}
test "Parse order of operations" {
    var parsed = try testParser("1 * 2 + 3 * 4");
    defer parsed.arena.deinit();
    const sum = parsed.expression.sum.items;
    try expect(2, sum.len);
    try expect(&[_]Expression{ Expression.newInteger(1), Expression.newInteger(2) }, sum[0].product.items);
    try expect(&[_]Expression{ Expression.newInteger(3), Expression.newInteger(4) }, sum[1].product.items);
}
test "Parse implicit multiplication" {
    var parsed = try testParser("2a^3");
    defer parsed.arena.deinit();
    const product = parsed.expression.product.items;
    try expect(2, product.len);
    try expect(Expression.newInteger(2), product[0]);
    const power = product[1].power;
    try expect(Expression.newVariable("a"), power[0]);
    try expect(Expression.newInteger(3), power[1]);
}
