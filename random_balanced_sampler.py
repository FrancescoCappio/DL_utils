import argparse
import random 




def generate_list(n_classes, sources_per_class, batch_p, lengths_dict):
    """
    n_classes -> total number of classes
    sources_per_class -> dict with {class_id : number of sources containing this class}
    batch_p -> number of samples for each class in a block
    lengths_dict -> number of samples in each class. Ex class 0 has K=sources_per_class[0] sources. lengths[0] = [len_class_0_source_1, ..., len_class_0_source_K] 
    ids_dict -> each sample has a unique identifier. This identifiers are those that should be inserted in the final list. This is a dict that contains for 
    each class and each souce 

    Should return a list which can be divided in blocks of size batch_p. Each block contains batch_p 
    elements of the same class. Subsequent blocks refer to different classes. 
    The sampling should always be done with replacement in order to maintain balancing. In particular

     - if for a certain class one source has less samples than the others those samples should be selected
     more often in order to rebalance the various sources;
     - if a certain class has in total a lower number of samples w.r.t. the others it should still appear in the 
     same number of blocks. 

     Therefore the correct approach is:

      - we compute the number of samples that we need for each class (max of each class number of sources*max_source_length)
      - for each class we randomly sample from the various sources (in an alternating fashion) until we reach the desired length


    Example of result with:
     - n_classes = 6
     - sources_per_class = {0:5,1:5,2:5,3:5,4:5,5:5} # -> each class has 5 sources
     - batch_p = 3
     - lengths_dict = {0:[8,8,8,8,8],1:[8,8,8,8,8],2:[8,8,8,8,8],3:[8,8,8,8,8],4:[8,8,8,8,8],5:[8,8,8,8,8]
     - indices_dict = {}

    OUTPUT: [
    D0C0E0, D1C0E0, D2C0E0,
    D0C1E0, D1C1E0, D2C1E0,
    D0C2E0, D1C2E0, D2C2E0,
    D0C3E0, D1C3E0, D2C3E0,
    D0C4E0, D1C4E0, D2C4E0,
    D0C5E0, D1C5E0, D2C5E0,

    D3C0E0, D4C0E0, D0C0E1,
    D3C1E0, D4C1E0, D0C1E1,
    D3C2E0, D4C2E0, D0C2E1,
    D3C3E0, D4C3E0, D0C3E1,
    D3C4E0, D4C4E0, D0C4E1,
    D3C5E0, D4C5E0, D0C5E1,

    D1C0E1, D2C0E1, D3C0E1,
    D1C1E1, D2C1E1, D3C1E1,
    D1C2E1, D2C2E1, D3C2E1,
    D1C3E1, D2C3E1, D3C3E1,
    D1C4E1, D2C4E1, D3C4E1,
    D1C5E1, D2C5E1, D3C5E1,

    D4C0E1, D0C0E2, D1C0E2,
    D4C1E1, D0C1E2, D1C1E2,
    D4C2E1, D0C2E2, D1C2E2,
    D4C3E1, D0C3E2, D1C3E2,
    D4C4E1, D0C4E2, D1C4E2,
    D4C5E1, D0C5E2, D1C5E2,

    ...
    ]

    First of all we compute the desired length for each class queue. 
    So for each class we compute num_sources*len_largest_source and we get the max of those values

    Then we create a queue for each class with the desired length and alternating samples from the various
    sources.

    We first create some intermediate parts that will help in finalizing the last list

    first for each source for each class we create a queue of elements:
    queue_C0_D0: [E0,E1,E2,E3,E4,E5,E6,E7]
    queue_C0_D1: [E0,E1,E2,E3,E4,E5,E6,E7]
    ...
    here we should take into account that sources should be balanced and therefore for those sources
    having a lower number of sample w.r.t. the others we will perform replacement

    Then for each class we create a queue that contains elements of that class alternating sources
    queue_C0 = [D0E0, D1E0, D2E0, D3E0, D4E0, D0E1, D1E1, D2E1, D3E1, D4E1, D0E2, D1E2, ...

    At this point we have a queue for each class. However it is possible that some queues are longer than others.
    Through resampling we should fix this so that we can keep the balancing between classes. 
    When resampling we should keep the alternating strategy for sources.

    """
    queues = {}

    for cls_id in range(n_classes):

        # first create a queue for each source
        src_qs = {}

        sources_this_class =sources_per_class[cls_id]

        for src in range(sources_this_class):
            # elements in this source:
            n_elem = lengths_dict[cls_id][src]
            src_qs[src] = [f'E{idx}' for idx in range(n_elem)]
            # now check if we should sample again to rebalance with larger source
            max_elem = max(lengths_dict[cls_id])
            while n_elem < max_elem:
                src_qs[source].append(random.randrange(n_elem))
                max_elem -= 1

        # now we put together sources 
        list_class = []
        while True: 
            found = False
            for src in range(sources_this_class):
                queue_this_source = src_qs[src]
                if len(queue_this_source) > 0:
                    found = True
                    list_class.append(f'D{src}{queue_this_source.pop(0)}')
            if not found:
                break
        queues[cls_id] = list_class

    out = []

    while True:
        found = False
        for cls_id in range(n_classes):
            q_this_class = queues[cls_id]
            if len(q_this_class) >= batch_p:
                found = True
                for el in range(batch_p):
                    out.append(f'C{cls_id}{q_this_class.pop(0)}')
        if not found:
            break
    return out
    

batch_p = 3
#sources_per_class = {0:5,1:5,2:5,3:5,4:5,5:5}
sources_per_class = {0:4,1:5,2:5,3:5,4:5,5:5}
#lengths_dict = {0:[8,8,8,8,8],1:[8,8,8,8,8],2:[8,8,8,8,8],3:[8,8,8,8,8],4:[8,8,8,8,8],5:[8,8,8,8,8]}
lengths_dict = {0:[8,8,8,8],1:[8,8,8,8,8],2:[8,8,8,8,8],3:[8,8,8,8,8],4:[8,8,8,8,8],5:[8,8,8,8,8]}

out = generate_list(n_classes=6, sources_per_class = sources_per_class, batch_p = batch_p, lengths_dict = lengths_dict)


for idx, el in enumerate(out):
    print(el,end=' ')
    if (idx+1)%batch_p == 0:
        print("")

