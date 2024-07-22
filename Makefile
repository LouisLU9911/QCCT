up:
	cd docker && docker-compose up -d
	docker ps

stop:
	cd docker && docker-compose stop

down: tc-drop
	cd docker && docker-compose down

build:
	bash scripts/build.sh

clean:
	rm -rf docker/client
	rm -rf docker/server

clean-all: clean
	cd custom-congestion-controller && cross clean
	cd echo && cross clean

show:
	@echo "================================================="
	ip addr
	@echo "================================================="
	ip tuntap show

tc: tc-drop
	cd scripts && bash apply_tc.sh

tc-drop:
	cd scripts && bash drop_tc.sh
