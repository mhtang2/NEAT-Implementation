from Net import Network
from Net.network import save_model, load_model
import random as rand


def run():
    Network.setParams(5+1,1,2)
    net = Network(5 + 1, 1, 2)
    for i in range(100):
        net.mutate_add_edge()
        net.mutate_add_node()

    inputs = [1238, 1238, 1230, 138201, 123]
    NUM_IT = 6
    outputs = []
    for i in range(NUM_IT):
        outputs.append(net.feedforward(inputs.copy())[0])
    save_model(net, "models/testnet.pkl")


    newModel = load_model("models/testnet.pkl")
    compareOutputs = [] 
    for i in range(NUM_IT):
        compareOutputs.append(newModel.feedforward(inputs.copy())[0])
    print(compareOutputs)
    print(outputs)
    assert(compareOutputs == outputs)