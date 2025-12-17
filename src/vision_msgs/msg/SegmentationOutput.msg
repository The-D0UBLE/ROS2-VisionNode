# SegmentationOutput.msg

# Metadata
string frame_id
builtin_interfaces/Time stamp

# Detection info
int32[] class_ids           # same order as boxes
float32[] scores
float32[] boxes             # flattened [x1, y1, x2, y2, x1, y1, x2, y2 ...] per object

# Optional: full masks
sensor_msgs/Image[] masks   # one mask per object, optional