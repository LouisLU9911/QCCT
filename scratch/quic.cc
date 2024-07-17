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

    //
    // We are interacting with the outside, real, world.  This means we have to 
    // interact in real-time and therefore means we have to use the real-time
    // simulator and take the time to calculate checksums.
    //
    GlobalValue::Bind ("SimulatorImplementationType", StringValue ("ns3::RealtimeSimulatorImpl"));
    GlobalValue::Bind ("ChecksumEnabled", BooleanValue (true));

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

    //
    // Use the TapBridgeHelper to connect to the pre-configured tap devices for 
    // the left side.  We go with "UseBridge" mode since the CSMA devices support
    // promiscuous mode and can therefore make it appear that the bridge is 
    // extended into ns-3.  The install method essentially bridges the specified
    // tap to the specified CSMA device.
    //
    NS_LOG_DEBUG("Setting up TapBridge to connect real applications...");
    TapBridgeHelper tapBridge;
    tapBridge.SetAttribute ("Mode", StringValue ("UseBridge"));
    tapBridge.SetAttribute ("DeviceName", StringValue ("tap-left"));
    tapBridge.Install (nodes.Get (0), devices.Get (0));
  
    //
    // Connect the right side tap to the right side CSMA device on the right-side
    // ghost node.
    //
    tapBridge.SetAttribute ("DeviceName", StringValue ("tap-right"));
    tapBridge.Install (nodes.Get (1), devices.Get (1));


    NS_LOG_DEBUG("Enabling pcap tracing...");
    // Enable pcap tracing
    pointToPoint.EnablePcapAll("quic-tap");

    //
    // Run the simulation for ten minutes to give the user time to play around
    //
    NS_LOG_DEBUG("Running simulation...");
    Simulator::Stop (Seconds (600.));
    Simulator::Run();
    Simulator::Destroy();

    NS_LOG_DEBUG("Simulation finished.");

    return 0;
}

