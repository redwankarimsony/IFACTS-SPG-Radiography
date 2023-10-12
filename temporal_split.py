import numpy as np

def temporal_sort(input_list):
    """This function sorts a list of filenames where AM should be before PM.
        At the same time for same category of AM or PM, the lower index should come first.

    Args:
        input_list (_type_): List if filenames like 
        ['1148_AM_Chest_AP15.txt', 
        '1148_AM_Chest_AP3.txt', 
        '1148_AM_Chest_AP2.txt', 
        '1148_AM_Chest_AP1.txt']

    Returns:
        _type_: Returns a temporally sorted list of filenames like:
        ['1148_AM_Chest_AP1.txt', 
        '1148_AM_Chest_AP2.txt', 
        '1148_AM_Chest_AP3.txt', 
        '1148_AM_Chest_AP15.txt']
    """


    AMs, PMs, idxAM, idxPM = [], [], [], []

    for inp_file in input_list:
        try:
            idx = int(float(inp_file.split("_")[3].replace("AP","").replace(".txt", "")))
        except:
            idx = 0
        
        if "AM" in inp_file: 
            AMs.append(inp_file)
            idxAM.append(idx)
        elif "PM" in inp_file: 
            PMs.append(inp_file)
            idxPM.append(idx)

    sorted_idx = np.argsort(idxAM)
    AM_sorted = [AMs[idx] for idx in sorted_idx]

    sorted_idx = np.argsort(idxPM)
    PM_sorted = [PMs[idx] for idx in sorted_idx]

    return AM_sorted + PM_sorted
        


if __name__ == "__main__":
    xxx = ['1148_AM_Chest_AP15.txt', '1148_AM_Chest_AP3.txt', '1148_AM_Chest_AP2.txt', 
        '1148_AM_Chest_AP1.txt']

    print(temporal_sort(xxx))


    
