up:
	cd docker && docker-compose up -d
	docker ps

stop:
	cd docker && docker-compose stop

down:
	cd docker && docker-compose down

build:
	bash docker/build.sh

clean:
	rm -rf docker/client
	rm -rf docker/server

clean-all: clean
	cd custom-congestion-controller && cross clean
	cd echo && cross clean
