import os


def create_experiment_folder(args):
    # Check if this is for testing
    if args.test:
        new_filename = str(args.test_folder)
        experiment_folder = os.path.join(args.result_folderpath, new_filename)
        args.result_folderpath = experiment_folder
        return new_filename

    # Get the name of the new folder for results
    filenames = [int(f) for f in os.listdir(args.result_folderpath)]
    new_filename = str(sorted(filenames)[-1] + 1) if len(filenames) else '0'

    # Create a folder for results and update args' result folder
    experiment_folder = os.path.join(args.result_folderpath, new_filename)
    os.makedirs(experiment_folder)
    args.result_folderpath = experiment_folder

    return new_filename
