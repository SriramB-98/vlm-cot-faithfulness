SCALE_FACTOR=0.25
MAX_PIXELS=$(echo "1003520 * $SCALE_FACTOR" | bc | cut -d. -f1)
echo '{"images_kwargs.do_resize":true, "images_kwargs.size.shortest_edge": 3136, "images_kwargs.size.longest_edge": '"$MAX_PIXELS"'}'




