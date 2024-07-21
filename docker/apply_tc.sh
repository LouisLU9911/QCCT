#!/bin/bash

# Get the container IDs
CONTAINER1_ID=$(docker ps -qf "name=container1")
CONTAINER2_ID=$(docker ps -qf "name=container2")

# Get the network interfaces (veth)
VETH1=$(docker exec $CONTAINER1_ID sh -c "cat /sys/class/net/eth0/iflink" | xargs -I{} find /sys/class/net -type l -name {} | xargs -I{} basename {})
VETH2=$(docker exec $CONTAINER2_ID sh -c "cat /sys/class/net/eth0/iflink" | xargs -I{} find /sys/class/net -type l -name {} | xargs -I{} basename {})

# Apply traffic control rules
# Add a delay of 100ms and a packet loss of 10%
tc qdisc add dev $VETH1 root netem delay 100ms loss 10%
tc qdisc add dev $VETH2 root netem delay 100ms loss 10%

echo "Traffic control applied to $VETH1 and $VETH2"

