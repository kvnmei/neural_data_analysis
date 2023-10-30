import itertools

# TODO: implement principal component regression

def run_kfold_cv(input_dict: dict, output_dict: dict, input_param_names: list[str]):
    # TODO: Finish implementing this... difficult roadblock because not sure input_dict keys are tuples
    #  and how long the tuple is and how to label them
    # noinspection GrazieInspection
    for input_key, output_key in itertools.product(
        input_dict.keys(), output_dict.keys()
    ):
        input_data = input_dict[input_key]
        output_data = output_dict[output_key]

        # print(f"Running {input_key}, {output_key}")
        # for i in range(len(input_key)):
        #     print(f"{}input_param_names[i]: {input_key[i]}")
        # print(
        #     f"Running  {neural_key[0]}, BRAIN AREA {neural_key[1]}, "
        #     f"BIN CENTER {neural_key[2]}, BIN SIZE {neural_key[3]}, IMAGE VARIABLE {image_key}"
        # )
        # {temp_list[i]: temp_tup[i] for i in range(len(temp_list))}
