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
sudo tc qdisc del dev $VETH1 root || echo "No tc rules for ${VETH1}"
sudo tc qdisc del dev $VETH2 root || echo "No tc rules for ${VETH2}"
