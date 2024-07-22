#!/bin/bash

# Get the container IDs
# server
CONTAINER1_ID=$(docker ps -qf "name=docker-quic-server-1")
# client
CONTAINER2_ID=$(docker ps -qf "name=docker-quic-client-1")

# Get the network interfaces (veth)
# VETH1=$(docker exec $CONTAINER1_ID sh -c "cat /sys/class/net/eth0/iflink" | xargs -I{} find /sys/class/net -type l -name {} | xargs -I{} basename {})
# VETH2=$(docker exec $CONTAINER2_ID sh -c "cat /sys/class/net/eth0/iflink" | xargs -I{} find /sys/class/net -type l -name {} | xargs -I{} basename {})
VETH_NO1=$(docker exec $CONTAINER1_ID sh -c "cat /sys/class/net/eth0/iflink")
VETH_NO2=$(docker exec $CONTAINER2_ID sh -c "cat /sys/class/net/eth0/iflink")

#VETH1=$(ip addr | grep "$VETH_NO1:" | awk '{print $2}' | awk '{sub(/:$/, ""); print}')
#VETH2=$(ip addr | grep "$VETH_NO2:" | awk '{print $2}' | awk '{sub(/:$/, ""); print}')

VETH1=$(ip addr | grep "$VETH_NO1:" | awk '{print $2}')
VETH2=$(ip addr | grep "$VETH_NO2:" | awk '{print $2}')

VETH1="${VETH1%@*}"
VETH2="${VETH2%@*}"

echo Server:$VETH1
echo Client:$VETH2

# Drop previous one
sudo tc qdisc del dev $VETH1 root || exit 1
sudo tc qdisc del dev $VETH2 root || exit 1

# Apply traffic control rules
# Add a delay of 100ms and a packet loss of 10%
sudo tc qdisc add dev $VETH1 root netem delay 100ms loss 10% || exit 1
sudo tc qdisc add dev $VETH2 root netem delay 100ms loss 10% || exit 1
echo "Traffic control applied to $VETH1 and $VETH2"
echo "======================================================="

sudo tc qdisc show dev $VETH1
sudo tc qdisc show dev $VETH2


