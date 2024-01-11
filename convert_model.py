import sys
import torch
import msgpack

def main():
	if len(sys.argv) != 3:
		print("Usage: python3 convert_model.py <path_to_model> <output_path>")
		return

	# if native is big endian, swap all the data later to save it in little endian
	swap = sys.byteorder == "big"

	model_path = sys.argv[1]
	output_path = sys.argv[2]

	output = open(output_path, "xb") # 'x' means to not overwrite if already exists, 'b' means writing bytes instead of text

	print("Loading the model weights...")
	model_weights = torch.load(model_path, map_location="cpu")

	# print(model_weights["tok_embeddings.weight"].float().numpy().shape)
	# print(model_weights["tok_embeddings.weight"][0][0])
	# return

	print("Model weights loaded. Converting...")

	for param in list(model_weights.keys()):
		weights = model_weights[param].float().numpy()
		if (weights.dtype.byteorder == "=" and swap) or weights.dtype.byteorder == ">":
			weights.byteswap()
			
		# (param_name, param_weights)
		msgpack.pack((param, weights.tobytes()), output)
		del model_weights[param] # to decrease RAM usage

	output.close()

	print("Success. Converted model weights:", output_path)


if __name__ == "__main__":
	main()
