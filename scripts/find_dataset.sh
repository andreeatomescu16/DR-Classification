#!/bin/bash
# Find where the dataset was downloaded

echo "Searching for dataset files..."
echo "================================"

# Check common locations
echo "Checking common locations:"
echo "  - data/combined_dataset: $([ -d "data/combined_dataset" ] && echo "EXISTS" || echo "NOT FOUND")"
echo "  - combined_dataset: $([ -d "combined_dataset" ] && echo "EXISTS" || echo "NOT FOUND")"
echo "  - data/: $([ -d "data" ] && echo "EXISTS" || echo "NOT FOUND")"

# Find directories with many images
echo ""
echo "Searching for directories with images..."
find . -type d -name "*eyepacs*" -o -name "*aptos*" -o -name "*train_images*" 2>/dev/null | head -10

# Find any large directories that might contain images
echo ""
echo "Checking for directories with many PNG/JPG files..."
for dir in data combined_dataset .; do
    if [ -d "$dir" ]; then
        png_count=$(find "$dir" -name "*.png" 2>/dev/null | wc -l)
        jpg_count=$(find "$dir" -name "*.jpg" 2>/dev/null | wc -l)
        if [ $png_count -gt 0 ] || [ $jpg_count -gt 0 ]; then
            echo "  $dir: $png_count PNG, $jpg_count JPG"
        fi
    fi
done

# Check what's in data/ if it exists
if [ -d "data" ]; then
    echo ""
    echo "Contents of data/:"
    ls -la data/ | head -20
fi
