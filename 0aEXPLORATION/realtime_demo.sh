#!/bin/bash

# Real-time YOLO processing simulation script
# Processes the sample image multiple times to simulate video frames

echo "🚀 Starting real-time YOLO processing simulation..."
echo "📱 Using GPU acceleration with CUDA"

# Create output directory
mkdir -p realtime_output
cd /home/rustuser/projects/rust/from_github/candle

# Number of frames to process
FRAMES=100
STATS_INTERVAL=10

echo "🎬 Processing $FRAMES frames..."
start_time=$(date +%s.%N)

for i in $(seq 1 $FRAMES); do
    # Process frame with YOLO v8 on GPU
    output_file="realtime_output/frame_$(printf '%04d' $i).jpg"
    
    # Copy original to new location, then process in-place
    cp candle-examples/examples/yolo-v8/assets/bike.jpg "$output_file"
    
    # Run YOLO detection (suppress output except for stats)
    if [ $((i % STATS_INTERVAL)) -eq 0 ]; then
        echo "📊 Processing frame $i/$FRAMES..."
        cargo run --example yolo-v8 --release --features cuda -- "$output_file" 2>/dev/null
        # Move the processed image to our output
        if [ -f bike.pp.jpg ]; then
            mv bike.pp.jpg "realtime_output/annotated_frame_$(printf '%04d' $i).jpg"
        fi
    else
        cargo run --example yolo-v8 --release --features cuda -- "$output_file" >/dev/null 2>&1
        if [ -f bike.pp.jpg ]; then
            mv bike.pp.jpg "realtime_output/annotated_frame_$(printf '%04d' $i).jpg"
        fi
    fi
    
    # Simulate real-time processing (remove for max speed)
    # sleep 0.033  # 30 FPS
done

end_time=$(date +%s.%N)
duration=$(echo "$end_time - $start_time" | bc)
fps=$(echo "scale=2; $FRAMES / $duration" | bc)

echo "✅ Processing complete!"
echo "📊 Stats:"
echo "   🎯 Frames processed: $FRAMES"
echo "   ⏱️  Total time: ${duration}s"
echo "   🚀 Average FPS: $fps"
echo "   📁 Output: realtime_output/"

echo ""
echo "🎥 To view results:"
echo "   ls realtime_output/annotated_frame_*.jpg"
echo "   # Or create a video:"
echo "   # ffmpeg -r 30 -i realtime_output/annotated_frame_%04d.jpg -c:v libx264 output_video.mp4"
