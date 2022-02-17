def category_indices(base_path, train_folder, gt_folder, train_prefix=None, gt_prefix=None, format=".txt"):
    
    '''
        This function assumes the following file structure:
        
        base_path
        -train_folder
            -train_prefix + {i} + .txt
        -gt_folder
            -gt_prefix + {i} + .txt
            
        And the following file contents:
        
        GT:
            a single integer corresponding to a category
        Train:
            N lines containing 3 comma-separated floats corresponding to point coordinates
    '''
    import os
    from tqdm import tqdm
    
    #
    train_prefix = train_prefix or train_folder 
    gt_prefix = gt_prefix or gt_folder 
    
    #
    train_path = os.path.join(base_path, train_folder)
    gt_path = os.path.join(base_path, gt_folder)
    
    #
    gt_file = lambda i : os.path.join(gt_path, gt_prefix + str(i) + format)
    train_file = lambda i : os.path.join(train_path, train_prefix + str(i) + format)
    
    #
    categories = {}
    
    #
    samples = os.listdir(train_path)
    N = len(samples)
    
    for i in tqdm(range(1, N+1)):
        with open(gt_file(i)) as GT:
            
            cat = GT.readline()[0]
            if cat not in categories.keys(): 
                categories[cat] = [i]
            else:
                categories[cat].append(i)
    
    save_path = os.path.join(base_path,"indices.txt");
    
    with open(save_path, "w") as F:
        for key in categories.keys():
            F.write(",".join(map(str, categories[key])) + "\n")

if __name__ == "__main__":

    #Ο φάκελος training περιέχει δύο φακέλους, έναν για τα δεδομένα και έναν για το gt
    #Βάλε το path αυτού του φακέλου και τρέξε την συνάρτηση. Μετά θα μπορείς να χρησιμοποιήσεις το dataset
    base = "C:\\Users\\vlassis\\Desktop\\phd\\datasets\\shrec2022\\training"
    category_indices(base, "pointCloud", "GTpointCloud")