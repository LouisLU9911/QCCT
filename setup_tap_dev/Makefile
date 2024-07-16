LEFT_NAME=tap-left
RIGHT_NAME=tap-right
THIS_USER=vboxuser

show:
	@echo "================================================="
	ip addr
	@echo "================================================="
	ip tuntap show

run: show
	@echo "================================================="
	sudo ip tuntap add dev $(LEFT_NAME) mode tap user $(THIS_USER)
	sudo ip tuntap add dev $(RIGHT_NAME) mode tap user $(THIS_USER)
	@echo "================================================="
	sudo ip link set dev $(LEFT_NAME) up
	sudo ip link set dev $(RIGHT_NAME) up
	@echo "================================================="
	sudo ip addr add 10.0.0.1/24 dev $(LEFT_NAME)
	sudo ip addr add 10.0.0.2/24 dev $(RIGHT_NAME)

clean: show
	sudo ip link delete $(LEFT_NAME)
	sudo ip link delete $(RIGHT_NAME)

