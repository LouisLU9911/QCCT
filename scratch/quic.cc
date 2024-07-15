#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/tap-bridge-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TapBridgeExample");

int main(int argc, char *argv[]) {
    CommandLine cmd;
    cmd.Parse(argc, argv);

    // Enable logging at the debug level for the entire script
    LogComponentEnable("TapBridgeExample", LOG_LEVEL_DEBUG);
    LogComponentEnable("PointToPointHelper", LOG_LEVEL_DEBUG);
    LogComponentEnable("TapBridgeHelper", LOG_LEVEL_DEBUG);

    NS_LOG_DEBUG("Creating nodes...");
    // Create nodes
    NodeContainer nodes;
    nodes.Create(2);

    NS_LOG_DEBUG("Setting up point-to-point link...");
    // Create point-to-point link
    PointToPointHelper pointToPoint;
    pointToPoint.SetDeviceAttribute("DataRate", StringValue("5Mbps"));
    pointToPoint.SetChannelAttribute("Delay", StringValue("2ms"));

    NetDeviceContainer devices;
    devices = pointToPoint.Install(nodes);

    NS_LOG_DEBUG("Installing internet stack...");
    // Install internet stack
    InternetStackHelper stack;
    stack.Install(nodes);

    NS_LOG_DEBUG("Assigning IP addresses...");
    // Assign IP addresses
    Ipv4AddressHelper address;
    address.SetBase("10.0.0.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    NS_LOG_DEBUG("Setting up TapBridge to connect real applications...");
    // Set up TapBridge to connect real applications
    TapBridgeHelper tapBridge1, tapBridge2;
    tapBridge1.SetAttribute("Mode", StringValue("UseLocal"));
    tapBridge1.SetAttribute("DeviceName", StringValue("tap-left"));
    tapBridge1.Install(nodes.Get(0), devices.Get(0));

    tapBridge2.SetAttribute("Mode", StringValue("UseLocal"));
    tapBridge2.SetAttribute("DeviceName", StringValue("tap-right"));
    tapBridge2.Install(nodes.Get(1), devices.Get(1));

    NS_LOG_DEBUG("Enabling pcap tracing...");
    // Enable pcap tracing
    pointToPoint.EnablePcapAll("ns3-tap");

    NS_LOG_DEBUG("Running simulation...");
    // Run simulation
    Simulator::Run();
    Simulator::Destroy();

    NS_LOG_DEBUG("Simulation finished.");

    return 0;
}

