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
    const TokenType = enum { Integer, Literal, Add, Subtract, Multiply, Divide, Power, Exclaim, OpenBracket, CloseBracket };

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
            '!' => return self.makeToken(TokenType.Exclaim),
            '0'...'9' => {
                while (!self.isDone() and '0' <= self.peek() and self.peek() <= '9') {
                    self.index += 1;
                }
                return self.makeToken(TokenType.Integer);
            },
            'a'...'z', 'A'...'Z', '_' => {
                if (!self.isDone() and self.peek() == '_') {
                    self.index += 1;
                    while (!self.isDone() and (('a' <= self.peek() and self.peek() <= 'z') or ('A' <= self.peek() and self.peek() <= 'Z') or self.peek() == '_')) {
                        self.index += 1;
                    }
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
    var tokeniser = Tokeniser.new("  aA_B  ");
    try expect((try tokeniser.next()).?.ty, Tokeniser.TokenType.Literal);
    try expect((try tokeniser.next()).?.ty, Tokeniser.TokenType.Literal);
    try expect((try tokeniser.next()), null);
}
test "Tokenise IntLit" {
    var tokeniser = Tokeniser.new("2a");
    try expect((try tokeniser.next()).?.ty, Tokeniser.TokenType.Integer);
    try expect((try tokeniser.next()).?.ty, Tokeniser.TokenType.Literal);
    try expect((try tokeniser.next()), null);
}
test "Tokenise Factorial" {
    var tokeniser = Tokeniser.new("2a!");
    try expect((try tokeniser.next()).?.ty, Tokeniser.TokenType.Integer);
    try expect((try tokeniser.next()).?.ty, Tokeniser.TokenType.Literal);
    try expect((try tokeniser.next()).?.ty, Tokeniser.TokenType.Exclaim);
    try expect((try tokeniser.next()), null);
}

/// Check if two lists of expressions are equal, ignoring the ordering
fn compare_unordered_expressions(alloc: std.mem.Allocator, a: *const std.ArrayListUnmanaged(Expression), b: *const std.ArrayListUnmanaged(Expression)) error{OutOfMemory}!bool {
    if (a.items.len != b.items.len) {
        return false;
    }
    var used_b: std.ArrayListUnmanaged(bool) = .empty;
    defer used_b.deinit(alloc);
    try used_b.appendNTimes(alloc, false, a.items.len);
    for (a.items) |needle| {
        var found = false;
        for (b.items, used_b.items) |haystack, *used| {
            if (used.*) {
                continue;
            }
            if (try haystack.equals(alloc, &needle)) {
                used.* = true;
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

const lcm = std.math.lcm;
const gcd = std.math.gcd;

const Numeric = struct {
    numerator: i32,
    denominator: u32,

    const Self = @This();
    pub fn add(self: Self, other: Self) Self {
        const denominator = lcm(self.denominator, other.denominator);
        const numerator = @as(i32, @intCast(@divExact(denominator, self.denominator))) * self.numerator + @as(i32, @intCast(@divExact(denominator, other.denominator))) * other.numerator;
        return (Self{ .denominator = denominator, .numerator = numerator }).simplified();
    }
    pub fn multiply(self: Self, other: Self) Self {
        const numerator = self.numerator * other.numerator;
        const denominator = self.denominator * other.denominator;
        return (Self{ .denominator = denominator, .numerator = numerator }).simplified();
    }
    /// Useful for displaying expressions nicely
    pub fn format(self: *const Self, writer: anytype) !void {
        if (self.denominator == 0) {
            try writer.print("DivZeroError", .{});
        } else if (self.denominator == 1) {
            try writer.print("{}", .{self.numerator});
        } else {
            try writer.print("({}/{})", .{ self.numerator, self.denominator });
        }
    }
    pub fn power(self: Self, exponent: i32) error{ Overflow, Underflow }!Self {
        var result = Self{ .numerator = try std.math.powi(i32, self.numerator, @intCast(@abs(exponent))), .denominator = try std.math.powi(u32, self.denominator, @abs(exponent)) };
        if (exponent < 0) {
            result = Self{ .numerator = std.math.sign(result.numerator) * @as(i32, @intCast(result.denominator)), .denominator = @abs(result.numerator) };
        }
        return result;
    }
    pub fn new(value: i32) Self {
        return Self{ .numerator = value, .denominator = 1 };
    }
    pub fn simplified(self: Self) Self {
        const divisor = gcd(@abs(self.denominator), @abs(self.numerator));

        return Self{
            .denominator = @abs(@divExact(self.denominator, divisor)),
            .numerator = @divExact(self.numerator, @as(i32, @intCast(divisor))),
        };
    }
    pub fn equals(self: Self, other: Self) bool {
        return self.simplified().numerator == other.simplified().numerator and self.simplified().denominator == other.simplified().denominator;
    }
    pub const ZERO = Self{ .numerator = 0, .denominator = 1 };
    pub const ONE = Self{ .numerator = 1, .denominator = 1 };
};

/// An expression is the equivilant of an abstract base class.
const Expression = union(enum) {
    integer: Numeric,
    variable: []const u8,
    product: std.ArrayListUnmanaged(Expression),
    sum: std.ArrayListUnmanaged(Expression),
    power: *[2]Expression,
    factorial: *Expression,

    const Self = @This();
    /// Useful for displaying expressions nicely
    pub fn format(self: *const Self, writer: anytype) !void {
        switch (self.*) {
            .integer => |value| try writer.print("{f}", .{value}),
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
            .factorial => |value| try writer.print("( {f} )!", .{value.*}),
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
        return Expression{ .integer = Numeric.new(value) };
    }
    /// Construct a numeric expression
    pub fn newNumeric(numeric: Numeric) Expression {
        return Expression{ .integer = numeric };
    }
    /// Construct a variable epxression
    pub fn newVariable(name: []const u8) Expression {
        return Expression{ .variable = name };
    }
    /// Negates the current expression by multiplying by -1
    fn negate(self: Self, alloc: std.mem.Allocator) std.mem.Allocator.Error!Expression {
        return Self.newProduct(&[_]Expression{ self, Expression.newInteger(-1) }, alloc);
    }
    /// Factorial
    fn newFactorial(self: Self, alloc: std.mem.Allocator) std.mem.Allocator.Error!Expression {
        const fact = try alloc.create(Expression);
        fact.* = self;
        return Expression{ .factorial = fact };
    }
    /// Reciprical of the current expression by rasining to the power of -1
    fn recip(self: Self, alloc: std.mem.Allocator) std.mem.Allocator.Error!Expression {
        return Self.newPower(self, Expression.newInteger(-1), alloc);
    }
    /// Does one expression equal another?
    fn equals(a: *const Self, alloc: std.mem.Allocator, b: *const Self) std.mem.Allocator.Error!bool {
        return switch (a.*) {
            .integer => |first| switch (b.*) {
                .integer => |second| first.equals(second),
                else => false,
            },
            .variable => |first| switch (b.*) {
                .variable => |second| std.mem.eql(u8, second, first),
                else => false,
            },
            .product => |first| switch (b.*) {
                .product => |second| return try compare_unordered_expressions(alloc, &first, &second),
                else => false,
            },
            .sum => |first| switch (b.*) {
                .sum => |second| return try compare_unordered_expressions(alloc, &first, &second),
                else => false,
            },
            .power => |first| switch (b.*) {
                .power => |second| try first[0].equals(alloc, &second[0]) and try first[1].equals(alloc, &second[1]),
                else => false,
            },
            .factorial => |first| switch (b.*) {
                .factorial => |second| try first.equals(alloc, second),
                else => false,
            },
        };
    }
    /// Collapse e.g. a sum of sums becomes a single sum
    fn collapsed_nested(self: *Self, alloc: std.mem.Allocator) error{ Overflow, Underflow, OutOfMemory }!void {
        switch (self.*) {
            .product => |*terms| {
                var index: usize = 0;
                while (index < terms.items.len) {
                    try terms.items[index].collapsed_nested(alloc);
                    switch (terms.items[index]) {
                        .product => |*product| try terms.replaceRange(alloc, index, 1, product.items),
                        .integer => |*integer| {
                            // One is identity
                            if (integer.equals(Numeric.ONE)) {
                                _ = terms.swapRemove(index);
                            } else if (integer.equals(Numeric.ZERO)) {
                                // Result is zero
                                self.* = Expression.newInteger(0);
                                return;
                            } else {
                                index += 1;
                            }
                        },
                        else => index += 1,
                    }
                }
                if (terms.items.len == 0) {
                    self.* = Expression.newInteger(1);
                } else if (terms.items.len == 1) {
                    self.* = terms.items[0];
                }
            },
            .sum => |*terms| {
                var index: usize = 0;
                while (index < terms.items.len) {
                    try terms.items[index].collapsed_nested(alloc);
                    switch (terms.items[index]) {
                        .sum => |*sum| try terms.replaceRange(alloc, index, 1, sum.items),
                        // Zero identity
                        .integer => |*integer| {
                            if (integer.equals(Numeric.ZERO)) {
                                _ = terms.swapRemove(index);
                            } else {
                                index += 1;
                            }
                        },
                        else => index += 1,
                    }
                }
                if (terms.items.len == 0) {
                    self.* = Expression.newInteger(0);
                } else if (terms.items.len == 1) {
                    self.* = terms.items[0];
                }
            },
            .power => |power| {
                try power[0].collapsed_nested(alloc);
                try power[1].collapsed_nested(alloc);
                switch (power[0]) {
                    .power => |inner| {
                        // Collapse nested powers
                        power[0] = inner[0];
                        power[1] = try Expression.newProduct(&[_]Expression{ inner[1], power[1] }, alloc);
                        try power[1].collapsed_nested(alloc);
                    },
                    .integer => |base| switch (power[1]) {
                        // Expand integer powers
                        .integer => |exponent| if (exponent.denominator == 1) {
                            self.* = Expression.newNumeric(try base.power(exponent.numerator));
                        },
                        else => {},
                    },
                    else => {},
                }
                switch (power[1]) {
                    .integer => |exponent| if (exponent.equals(Numeric.ONE)) {
                        // Simplify power of 1
                        self.* = power[0];
                    } else if (exponent.equals(Numeric.ZERO)) {
                        // Simplify power of 0
                        self.* = Expression.newInteger(1);
                    },
                    else => {},
                }
            },
            .factorial => |value| {
                try value.collapsed_nested(alloc);
                switch (value.*) {
                    .integer => |max| {
                        if (max.denominator == 1) {
                            var result: i32 = 1;
                            var item: i32 = 1;
                            while (item <= max.numerator) {
                                result *= item;
                                item += 1;
                            }
                            self.* = Expression.newInteger(result);
                        }
                    },
                    else => {},
                }
            },

            .integer, .variable => {},
        }
    }
    fn without_multiple(self: *Self, alloc: std.mem.Allocator) error{ OutOfMemory, Overflow, Underflow }!Numeric {
        switch (self.*) {
            .product => |*p| {
                var count = Numeric.ONE;
                var index: usize = 0;
                while (index < p.items.len) {
                    switch (p.items[index]) {
                        .integer => |i| {
                            count = count.multiply(i);
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
            else => return Numeric.ONE,
        }
    }
    fn without_power(self: *Self) Numeric {
        switch (self.*) {
            .power => |p| {
                switch (p[1]) {
                    .integer => |i| {
                        self.* = p[0];
                        return i;
                    },
                    else => return Numeric.ONE,
                }
            },
            else => return Numeric.ONE,
        }
    }

    /// Expand (x+1)(x-1)
    fn expand_brakets(self: *Self, alloc: std.mem.Allocator) error{ OutOfMemory, Overflow, Underflow }!void {
        switch (self.*) {
            .sum => |terms| for (terms.items) |*term| {
                try term.expand_brakets(alloc);
            },
            .product => |terms| {
                var expanded: std.ArrayListUnmanaged(Expression) = .empty;
                var sum_indexes: std.ArrayListUnmanaged(usize) = .empty;
                try sum_indexes.appendNTimes(alloc, 0, terms.items.len);
                while (true) {
                    var got_new = false;
                    var next_term: std.ArrayListUnmanaged(Expression) = .empty;
                    try next_term.ensureTotalCapacityPrecise(alloc, terms.items.len);
                    for (terms.items, sum_indexes.items, 0..) |term, sum_index, index| {
                        switch (term) {
                            .sum => |sum| {
                                next_term.appendAssumeCapacity(sum.items[sum_index]);
                                if (!got_new and sum_index < sum.items.len - 1) {
                                    got_new = true;
                                    // Reset all previous sum indexes
                                    for (sum_indexes.items[0..index]) |*item| {
                                        item.* = 0;
                                    }
                                    sum_indexes.items[index] += 1;
                                }
                            },
                            else => {
                                next_term.appendAssumeCapacity(term);
                            },
                        }
                    }
                    try expanded.append(alloc, Expression{ .product = next_term });
                    if (!got_new) {
                        break;
                    }
                }
                self.* = Expression{ .sum = expanded };
            },
            .power => |values| {
                switch (values[1]) {
                    // Expand integer powers
                    .integer => |i| if (i.denominator == 1 and i.numerator > 0 and i.numerator < 6) {
                        var list: std.ArrayListUnmanaged(Expression) = .empty;
                        try list.appendNTimes(alloc, values[0], @intCast(i.numerator));
                        self.* = Expression{ .product = list };

                        try self.expand_brakets(alloc);
                    },
                    else => {
                        try values[0].expand_brakets(alloc);
                        try values[1].expand_brakets(alloc);
                    },
                }
            },
            .factorial => |value| try value.*.expand_brakets(alloc),
            .integer, .variable => {},
        }
    }

    /// Collect like terms e.g. x + 2 + 2x becomes 3x + 2
    fn collect_like_terms(self: *Self, alloc: std.mem.Allocator) error{ OutOfMemory, Overflow, Underflow }!void {
        switch (self.*) {
            .sum => |*oldTerms| {
                var newTerms: std.ArrayListUnmanaged(Expression) = .empty;
                for (oldTerms.items) |*oldTerm| {
                    try oldTerm.collect_like_terms(alloc);
                    const oldMultiple = try oldTerm.without_multiple(alloc);
                    var canConsolidate = false;
                    for (newTerms.items) |*newTerm| {
                        var new = try newTerm.without_multiple(alloc);
                        try newTerm.collapsed_nested(alloc);
                        canConsolidate = try newTerm.equals(alloc, oldTerm);
                        if (canConsolidate) {
                            new = new.add(oldMultiple);
                        }
                        newTerm.* = try Expression.newProduct(&[_]Expression{ Expression.newNumeric(new), newTerm.* }, alloc);
                        try newTerm.collapsed_nested(alloc);
                        if (canConsolidate) {
                            break;
                        }
                    }
                    if (!canConsolidate) {
                        var value = try Expression.newProduct(&[_]Expression{ Expression.newNumeric(oldMultiple), oldTerm.* }, alloc);
                        try value.collapsed_nested(alloc);
                        try newTerms.append(alloc, value);
                    }
                }
                oldTerms.* = newTerms;
            },
            .product => |*oldTerms| {
                var newTerms: std.ArrayListUnmanaged(Expression) = .empty;
                var multiple = Numeric.ONE;
                for (oldTerms.items) |*oldTerm| {
                    try oldTerm.collect_like_terms(alloc);

                    // Raw numeric literals should just be multiples
                    switch (oldTerm.*) {
                        .integer => |i| {
                            multiple = multiple.multiply(i);
                            continue;
                        },
                        else => {},
                    }

                    const oldPower = oldTerm.without_power();
                    var canConsolidate = false;
                    for (newTerms.items) |*newTerm| {
                        var newPowers = newTerm.without_power();
                        try newTerm.collapsed_nested(alloc);
                        canConsolidate = try newTerm.equals(alloc, oldTerm);
                        if (canConsolidate) {
                            newPowers = newPowers.add(oldPower);
                        }
                        newTerm.* = try Expression.newPower(newTerm.*, Expression.newNumeric(newPowers), alloc);
                        try newTerm.collapsed_nested(alloc);

                        if (canConsolidate) {
                            break;
                        }
                    }
                    if (!canConsolidate) {
                        var value = try Expression.newPower(oldTerm.*, Expression.newNumeric(oldPower), alloc);
                        try value.collapsed_nested(alloc);
                        try newTerms.append(alloc, value);
                    }
                }
                if (!multiple.equals(Numeric.ONE)) {
                    try newTerms.insert(alloc, 0, Expression.newNumeric(multiple));
                }
                oldTerms.* = newTerms;
            },
            .power => |values| {
                try values[0].collect_like_terms(alloc);
                try values[1].collect_like_terms(alloc);
            },
            .factorial => |value| try value.collect_like_terms(alloc),
            .integer, .variable => {},
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
            if (current.ty == TokenType.Exclaim) {
                expression = try expression.newFactorial(self.allocator());
                _ = try self.advance();
                continue;
            }
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
        try result.expression.expand_brakets(result.arena.allocator());
        try result.expression.collect_like_terms(result.arena.allocator());
        return result;
    }
};

/// A utilty for testing the parser
fn testParser(source: []const u8) !Parser.ParseResult {
    var parser = try Parser.new(std.testing.allocator, source);
    const parsed = try parser.parse();
    print("source: {s: <15}parsed: {f}\n", .{ source, parsed.expression });
    return parsed;
}

/// A utilty for testing the parser
fn testSimplified(source: []const u8) !Parser.ParseResult {
    var parser = try Parser.new(std.testing.allocator, source);
    const result = try parser.parse_and_simplify();
    print("source: {s: <15}parsed & simplified: {f}\n", .{ source, result.expression });
    return result;
}
fn expr_equals(expr: *const Expression, expected: []const u8) !void {
    var parser = try Parser.new(std.testing.allocator, expected);
    var parsed = try parser.parse();
    try parsed.expression.collect_like_terms(parsed.arena.allocator());
    const is_equal = try expr.equals(std.testing.allocator, &parsed.expression);
    defer parsed.arena.deinit();
    if (!is_equal) {
        print("{s}\nFAIL:\n  source:\t{f}\n  expected:\t{f}\n", .{ "-" ** 30, expr, parsed.expression });
        return error.ExprNotEq;
    }
}
fn simplified_source_eq(source: []const u8, expected: []const u8) !void {
    var parsed = try testSimplified(source);
    defer parsed.arena.deinit();
    try expr_equals(&parsed.expression, expected);
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
    var parsed = try testParser("7 * 2 + 3 * 4");
    defer parsed.arena.deinit();
    const sum = parsed.expression.sum.items;
    try expect(2, sum.len);
    try expect(&[_]Expression{ Expression.newInteger(7), Expression.newInteger(2) }, sum[0].product.items);
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
test "Equals" {
    try simplified_source_eq("3", "3");
    try simplified_source_eq("a", "a");
    try simplified_source_eq("a3", "3a");
    try simplified_source_eq("2+3a", "3a+2");
    try simplified_source_eq("2-b+3a", "-b+3a+2");

    try std.testing.expectError(error.ExprNotEq, simplified_source_eq("3 a b", "3a"));
    try std.testing.expectError(error.ExprNotEq, simplified_source_eq("3+a", "3+a+b"));
}
test "Simplify add" {
    try simplified_source_eq("2a+a", "3a");
    try simplified_source_eq("a+a", "2a");
    try simplified_source_eq("a-a", "0");
    try simplified_source_eq("a+b+2a+5b", "3a+6b");
    try simplified_source_eq("a/3 - a/4", "a/12");
}
test "Simplify Factorial" {
    try simplified_source_eq("4a!!", "4((a!)!)");
    try std.testing.expectError(error.ExprNotEq, simplified_source_eq("4a!", "(4a)!"));
}
test "Simplify product" {
    try simplified_source_eq("2a*a*a", "2a^3");
    try simplified_source_eq("2a*a*a+2b*b^4", "2a^3+2b^5");
    try simplified_source_eq("a a", "a^2");
    try simplified_source_eq("a b a b", "a^2 b^2");
    try simplified_source_eq("a / a", "1");
    try simplified_source_eq("a / a + b/b", "2");
}
test "Expand" {
    try simplified_source_eq("(x+1)(x-1)", "x^2-1");
    try simplified_source_eq("(x-1)^4", "x^4 + -4x^3 + 6x^2 + -4x + 1");
    try simplified_source_eq("(a+b+c)(d+e+f)(g+h+i)", "a*d*g + b*d*g + c*d*g + a*e*g + b*e*g + c*e*g + a*f*g + b*f*g + c*f*g + a*d*h + b*d*h + c*d*h + a*e*h + b*e*h + c*e*h + a*f*h + b*f*h + c*f*h + a*d*i + b*d*i + c*d*i + a*e*i + b*e*i + c*e*i + a*f*i + b*f*i + c*f*i");
}
