from dpp.data.loaders import InteractingPointsDataLoader, ConditionalWasssersteingPointDataLoader
from dpp.utils.datahandling import interacting_systems_load_data

basic_event_data_loader =  {"data_path": "/home/an/Desktop/Projects/PointProcesses/Results/NonlinearHawkes/InteractingPointsData/",
                            "batch_size": 32,
                             "bptt_size": 100,
                             "shuffle": True,
                             "num_workers": 1}

coditional_wasserstein = {"data_path": "/home/an/Desktop/Projects/PointProcesses/Results/NonlinearHawkes/InteractingPointsData/",
                          "batch_size": 32,
                          "past_of_sequence": .7,
                          "shuffle": True,
                          "num_workers": 1}

if __name__=="__main__":
    #POINT PROCESSES
    print("Point Processes")
    data_loader = InteractingPointsDataLoader(**basic_event_data_loader)
    for batch_idx, (x,relations) in enumerate(data_loader.train):
        print(x.shape)
        print(relations[0])
        print(relations[1])
        break

    #CONDITIONAL WASSERSTEIN
    print("Conditional Wasserstein Point Processes")
    data_loader = ConditionalWasssersteingPointDataLoader(**coditional_wasserstein)
    for batch_idx, (past,future) in enumerate(data_loader.train):
        print(past.shape)
        print(future.shape)
        break

    #INTERACTING SYSTEMS
    print(" Interacting Systems")
    data_dir = "/home/an/Desktop/Projects/General/interacting-systems/data/"
    suffix = '_springs5'
    train_data_loader, valid_data_loader, test_data_loader, loc_max, loc_min, vel_max, vel_min = interacting_systems_load_data(data_dir,10,suffix)
    for batch_idx, (data, relations) in enumerate(train_data_loader):
        print("data")
        print(data.shape)
        print("relations")
        print(relations.shape)
        break


