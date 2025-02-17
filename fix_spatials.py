import yaml
import math
import sys

def read_yaml(filename):
    """Reads a YAML file and returns its content."""
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
        return data[0] if isinstance(data, list) else data  # Extract first element if list

def write_yaml(filename, data):
    """Writes updated content back to the YAML file."""
    with open(filename, 'w') as file:
        yaml.safe_dump(data, file)  # Keep format consistent with lists

def extract_dims(data):
    """Extracts 'Dnumber' nodes and their lists from 'spatial_mapping_hint'."""
    dims = {}
    spatial_mapping = data.get("spatial_mapping_hint", {})
    for key, value in spatial_mapping.items():
        if key.startswith("D") and isinstance(value, list):
            dims[key] = value
    return dims

def extract_sizes(data):
    """Extracts loop dimensions and their sizes from 'loop_dims' and 'loop_sizes'."""
    loop_dims = data.get("loop_dims", [])
    loop_sizes = data.get("loop_sizes", [])
    return {dim: size for dim, size in zip(loop_dims, loop_sizes)}

def compute_new_sizes(sizes, dimensions, dims, szs):
    """Computes the largest divisor for each size that meets the given conditions."""
    new_sizes = []
    for S, D in zip(sizes, dimensions):
        if D not in dims:
            new_sizes.append(S)
            continue
        
        # Compute the product of sizes for the dimensions
        prod_val = math.prod(szs[d] for d in dims[D] if d in szs)
        
        # Find the largest divisor ≤ S
        valid_values = [x for x in range(1, S + 1) if prod_val % x == 0]
        
        new_sizes.append(max(valid_values) if valid_values else S)  # Fallback if no valid divisor

    return new_sizes

def main(file_a, file_b, file_c, file_out):
    # Step 1: Read and extract dims
    data_a = read_yaml(file_a)
    dims = extract_dims(data_a)

    # Step 2: Read and extract loop sizes
    data_b = read_yaml(file_b)
    szs = extract_sizes(data_b)

    # Step 3: Read, compute, and update sizes in file C
    data_c = read_yaml(file_c)

    operational_array = data_c.get("operational_array", {})

    if "sizes" in operational_array and "dimensions" in operational_array:
        sizes = operational_array["sizes"]
        dimensions = operational_array["dimensions"]
        new_sizes = compute_new_sizes(sizes, dimensions, dims, szs)

        # Update values
        operational_array["sizes"] = new_sizes
        write_yaml(file_out, data_c)
        print(f"Updated {file_c} successfully.")
    else:
        print("Invalid structure in file C. Missing 'sizes' or 'dimensions' under 'operational_array'.")


if __name__ == "__main__":
    assert len(sys.argv) == 3
    arch = sys.argv[1]
    comp = sys.argv[2]
    main(f"zigzag/inputs/mapping/{arch}_like_conv.yaml", f"emjzero/{comp}.yaml", f"zigzag/inputs/hardware/{arch}_like.yaml", "zigzag/inputs/hardware/tmp.yaml")
