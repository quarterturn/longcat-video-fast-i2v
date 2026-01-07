# longcat-video-fast-i2v
image-to-video script for longcat-video-fast

A script to provide i2v long video gen using the code on https://github.com/xlite-dev/longcat-video-fast
Be sure to install all the requirements listed there first, then copy this script into the root of the repo.

Usage example:
LONGCAT_VIDEO_DIR=LongCat-Video/weights/LongCat-Video torchrun --nproc_per_node=1 longcat_i2v_fast.py --compile  --image_path ./LongCat-Video/assets/_eecb8594-162f-45f1-b6ce-b552d640a904.jpg  --prompt "An adorable blonde-haired, hazel-eyed young woman leads the viewer out onto a disco dancefloor, as the lights go down, the mirror ball illuminates and casts beams of light through the smoky air, and the dance floor pulses with colordful ssquares, and the young woman begins to express her energy and desire as she dances to the rhythm of the music" --num_segments 10

You will need 48GB of VRAM to run the above example at fp8. If you have 32GB, it will switch to nf4, but I found going that low really hurts the model in terms of motion vectors and 3D comprehension.
Test with a low --num_segments, like 2, before committing to a long gen.
