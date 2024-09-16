// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/// Example implementation of a custom congestion controller algorithm.
///
/// This example serves only to illustrate the integration points for incorporating a custom
/// congestion controller into s2n-quic, and not as an actual congestion controller implementation.
///
/// NOTE: The `CongestionController` trait is considered unstable and may be subject to change
///       in a future release.
pub mod custom_congestion_controller {
    use s2n_quic::provider::{
        congestion_controller,
        congestion_controller::{
            CongestionController, Publisher, RandomGenerator, RttEstimator, Timestamp,
        },
    };
    use std::fmt;
    use std::sync::Arc;
    use tch::{kind, Kind, Tensor};
    use tch::{no_grad, CModule, Device};

    pub struct FifoQueue {
        buffer: Vec<Vec<f32>>, // Vec to hold events, each event is a Vec<f32>
        capacity: usize,
        head: usize,
        tail: usize,
        size: usize,
    }

    impl FifoQueue {
        // Create a new FIFO queue with a given capacity
        pub fn new(capacity: usize) -> Self {
            FifoQueue {
                buffer: vec![vec![0.0; 8]; capacity],
                capacity,
                head: 0,
                tail: 0,
                size: capacity,
            }
        }

        // Add an event to the queue
        pub fn enqueue(&mut self, event: Vec<f32>) {
            if self.size == self.capacity {
                // Overwrite the oldest event if the buffer is full
                self.buffer[self.tail] = event;
                self.tail = (self.tail + 1) % self.capacity;
                self.head = self.tail; // Move head when overwriting
            } else {
                self.buffer[self.tail] = event;
                self.tail = (self.tail + 1) % self.capacity;
                self.size += 1;
            }
        }

        // Convert the entire queue to a Tensor
        pub fn to_tensor(&self) -> Tensor {
            let events: Vec<f32> = self
                .buffer
                .iter()
                .take(self.size) // Only take the filled part of the buffer
                .flatten() // Flatten Vec<Vec<f32>> into Vec<f32>
                .cloned() // Clone the values to avoid borrowing issues
                .collect();

            let num_features = if self.size > 0 {
                self.buffer[0].len()
            } else {
                0
            };

            Tensor::from_slice(&events).view((self.size as i64, num_features as i64))
        }

        pub fn modify_last_event_final_item(&mut self, new_value: f32) {
            if self.size > 0 {
                // Calculate the index of the last event
                let last_index = if self.tail == 0 {
                    self.capacity - 1
                } else {
                    self.tail - 1
                };

                // Modify the last element of the last event
                if let Some(last_event) = self.buffer.get_mut(last_index) {
                    if let Some(last_element) = last_event.last_mut() {
                        *last_element = new_value;
                    }
                }
            }
        }
    }

    impl fmt::Debug for FifoQueue {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.debug_struct("FifoQueue")
                .field("buffer", &self.buffer)
                .field("capacity", &self.capacity)
                .field("head", &self.head)
                .field("tail", &self.tail)
                .field("size", &self.size)
                .finish()
        }
    }

    impl Clone for FifoQueue {
        fn clone(&self) -> Self {
            FifoQueue {
                buffer: self.buffer.clone(), // Clone the buffer Vec
                capacity: self.capacity,     // usize is Copy, so no need for `.clone()`
                head: self.head,             // usize is Copy
                tail: self.tail,             // usize is Copy
                size: self.size,             // usize is Copy
            }
        }
    }

    struct InferenceEngine {
        model: Arc<CModule>,
        model_name: String,
    }

    impl Clone for InferenceEngine {
        fn clone(&self) -> Self {
            InferenceEngine {
                model: Arc::clone(&self.model),
                model_name: self.model_name.clone(),
            }
        }
    }

    impl fmt::Debug for InferenceEngine {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("InferenceEngine")
                .field("model_name", &self.model_name)
                .finish()
        }
    }

    impl InferenceEngine {
        fn new(model_path: &str, device: Device) -> Self {
            let model =
                CModule::load_on_device(model_path, device).expect("Failed to load the model");

            // Extract the model's name from the path (for simplicity, using the filename)
            let model_name = std::path::Path::new(model_path)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("Unknown Model")
                .to_string();

            InferenceEngine {
                model: Arc::new(model),
                model_name,
            }
        }

        fn run_inference(&self, input_tensor: Tensor) -> Tensor {
            no_grad(|| self.model.forward_ts(&[input_tensor])).expect("Failed to run inference")
        }
    }

    /// Define a congestion controller containing any state you wish to track.
    /// For this example, we track the size of the congestion window in bytes and
    /// the number of bytes in flight.
    #[derive(Debug, Clone)]
    pub struct MyCongestionController {
        congestion_window: u32,
        bytes_in_flight: u32,
        events: FifoQueue,
        inference_engine: InferenceEngine,
    }

    /// The following is a simple implementation of the `CongestionController` trait
    /// that increases the congestion window by the number of bytes acknowledged and
    /// decreases the congestion window by half when packets are lost.
    #[allow(unused)]
    impl CongestionController for MyCongestionController {
        // A custom `PacketInfo` type may optionally be defined to include additional per-packet
        // state. This state will be stored for each in flight packet, and returned to the
        // `on_ack` and `on_packet_lost` methods.
        type PacketInfo = ();

        fn congestion_window(&self) -> u32 {
            self.congestion_window
        }

        fn bytes_in_flight(&self) -> u32 {
            self.bytes_in_flight
        }

        fn is_congestion_limited(&self) -> bool {
            self.congestion_window < self.bytes_in_flight
        }

        fn requires_fast_retransmission(&self) -> bool {
            false
        }

        fn on_packet_sent<Pub: Publisher>(
            &mut self,
            time_sent: Timestamp,
            sent_bytes: usize,
            app_limited: Option<bool>,
            rtt_estimator: &RttEstimator,
            publisher: &mut Pub,
        ) -> Self::PacketInfo {
            self.bytes_in_flight += sent_bytes as u32;
            // timestamp,lost_bytes,bytes_acknowledged,bytes_in_flight,event_on_ack,event_on_packet_lost,event_on_packet_sent,congestion_window
            let ms = time_sent.to_ms();
            let bytes_in_flight_f32 = self.bytes_in_flight as f32;
            let event: Vec<f32> = Vec::from([ms, 0.0, 0.0, bytes_in_flight_f32, 0.0, 0.0, 1.0, 0.0]);
            self.events.enqueue(event);
            let input = self.events.to_tensor().unsqueeze(0);
            let output = self.inference_engine.run_inference(input);
            let next_cwnd = output.double_value(&[0, 0]) as f32;
            self.events.modify_last_event_final_item(next_cwnd);
            self.congestion_window = next_cwnd as u32;
            // eprintln!("on_packet_sent: {next_cwnd}");
        }

        fn on_rtt_update<Pub: Publisher>(
            &mut self,
            time_sent: Timestamp,
            now: Timestamp,
            rtt_estimator: &RttEstimator,
            publisher: &mut Pub,
        ) {
            // no op
        }

        fn on_ack<Pub: Publisher>(
            &mut self,
            newest_acked_time_sent: Timestamp,
            bytes_acknowledged: usize,
            newest_acked_packet_info: Self::PacketInfo,
            rtt_estimator: &RttEstimator,
            random_generator: &mut dyn RandomGenerator,
            ack_receive_time: Timestamp,
            publisher: &mut Pub,
        ) {
            self.bytes_in_flight -= bytes_acknowledged as u32;
            // timestamp,lost_bytes,bytes_acknowledged,bytes_in_flight,event_on_ack,event_on_packet_lost,event_on_packet_sent,congestion_window
            let ms = ack_receive_time.to_ms();
            let bytes_in_flight_f32 = self.bytes_in_flight as f32;
            let bytes_acknowledged_f32 = bytes_acknowledged as f32;
            let event: Vec<f32> = Vec::from([ms, 0.0, bytes_acknowledged_f32, bytes_in_flight_f32, 1.0, 0.0, 0.0, 0.0]);
            self.events.enqueue(event);
            let input = self.events.to_tensor().unsqueeze(0);
            let output = self.inference_engine.run_inference(input);
            let next_cwnd = output.double_value(&[0, 0]) as f32;
            self.events.modify_last_event_final_item(next_cwnd);
            self.congestion_window = next_cwnd as u32;
            // eprintln!("on_ack: {next_cwnd}");
        }

        fn on_packet_lost<Pub: Publisher>(
            &mut self,
            lost_bytes: u32,
            packet_info: Self::PacketInfo,
            persistent_congestion: bool,
            new_loss_burst: bool,
            random_generator: &mut dyn RandomGenerator,
            timestamp: Timestamp,
            publisher: &mut Pub,
        ) {
            // Loss-based congestion controllers such as New Reno or CUBIC will decrease the
            // congestion window when a packet is lost. In this simple example congestion
            // controller the congestion window is reduced for every packet; an actual congestion
            // controller should take a more nuanced approach. This reduction would typically only
            // occur once for the initial lost packet, and subsequent lost packets would not lead to
            // further reduction.
            self.bytes_in_flight -= lost_bytes;
            // timestamp,lost_bytes,bytes_acknowledged,bytes_in_flight,event_on_ack,event_on_packet_lost,event_on_packet_sent,congestion_window
            let ms = timestamp.to_ms();
            let bytes_in_flight_f32 = self.bytes_in_flight as f32;
            let lost_bytes_f32 = lost_bytes as f32;
            let event: Vec<f32> = Vec::from([ms, lost_bytes_f32, 0.0, bytes_in_flight_f32, 0.0, 1.0, 0.0, 0.0]);
            self.events.enqueue(event);
            let input = self.events.to_tensor().unsqueeze(0);
            let output = self.inference_engine.run_inference(input);
            let next_cwnd = output.double_value(&[0, 0]) as f32;
            self.events.modify_last_event_final_item(next_cwnd);
            self.congestion_window = next_cwnd as u32;
            // eprintln!("on_packet_lost: {next_cwnd}");
        }

        fn on_explicit_congestion<Pub: Publisher>(
            &mut self,
            ce_count: u64,
            event_time: Timestamp,
            publisher: &mut Pub,
        ) {
            self.congestion_window = (self.congestion_window as f32 * 0.5) as u32;
        }

        fn on_mtu_update<Pub: Publisher>(&mut self, max_data_size: u16, publisher: &mut Pub) {
            // no op
        }

        fn on_packet_discarded<Pub: Publisher>(&mut self, bytes_sent: usize, publisher: &mut Pub) {
            self.bytes_in_flight -= bytes_sent as u32;
        }

        fn earliest_departure_time(&self) -> Option<Timestamp> {
            None
        }
    }

    // Define an endpoint for the custom congestion controller so it may be used as a
    // congestion controller provider to the s2n-quic server or client
    #[derive(Debug, Default)]
    pub struct MyCongestionControllerEndpoint {}

    impl congestion_controller::Endpoint for MyCongestionControllerEndpoint {
        type CongestionController = MyCongestionController;

        // This method will be called whenever a new congestion controller instance is needed.
        fn new_congestion_controller(
            &mut self,
            path_info: congestion_controller::PathInfo,
        ) -> Self::CongestionController {
            let model_path = "model_cpu.pt";
            let context_size = 32;

            let events = FifoQueue::new(context_size);
            let inference_engine = InferenceEngine::new(model_path, Device::Cpu);
            MyCongestionController {
                // Specify the initial congestion window
                congestion_window: 10 * path_info.max_datagram_size as u32,
                bytes_in_flight: 0,
                events,
                inference_engine,
            }
        }
    }
}
