#!/usr/bin/env python3
"""
Script to fix basicsr torchvision compatibility issue.
Replaces:
    from torchvision.transforms.functional_tensor import rgb_to_grayscale
with:
    from torchvision.transforms.functional import rgb_to_grayscale
in basicsr/data/degradations.py
"""

import sys
import os

def fix_basicsr_import():
    """Fix the torchvision compatibility issue in basicsr."""
    try:
        import basicsr
        basicsr_path = os.path.dirname(basicsr.__file__)
        degradations_file = os.path.join(basicsr_path, 'data', 'degradations.py')
        
        if not os.path.exists(degradations_file):
            print(f"Error: Could not find {degradations_file}")
            return False
        
        # Read the file
        with open(degradations_file, 'r') as f:
            content = f.read()
        
        # Check if the old import exists
        old_import = 'from torchvision.transforms.functional_tensor import rgb_to_grayscale'
        new_import = 'from torchvision.transforms.functional import rgb_to_grayscale'
        
        if old_import in content:
            # Replace the import
            content = content.replace(old_import, new_import)
            
            # Write the file back
            with open(degradations_file, 'w') as f:
                f.write(content)
            
            print(f"Successfully fixed {degradations_file}")
            return True
        else:
            print(f"Import already fixed or not found in {degradations_file}")
            return True
            
    except ImportError:
        print("basicsr is not installed yet. Skipping fix.")
        return True
    except Exception as e:
        print(f"Error fixing basicsr: {e}")
        return False

if __name__ == "__main__":
    success = fix_basicsr_import()
    sys.exit(0 if success else 1)