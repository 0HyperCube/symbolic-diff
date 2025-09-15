cli: build-cli
	./cli

web: build-web
	python3 -m http.server 8080 -d web

test:
	zig test parser.zig

clean:
	rm -f *.o *.wasm cli

build-cli:
	zig build-exe cli.zig -freference-trace=99

build-web:
	zig build-exe web.zig -target wasm32-freestanding -fno-entry --export=eval --export=free --export=alloc -O ReleaseSmall  -freference-trace=99 -femit-bin=web/origional_build.wasm
	wasm-opt -O4 web/origional_build.wasm -o web/web.wasm --enable-bulk-memory-opt


