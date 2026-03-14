# Simple carbon emissions estimator

transport_emissions = {
    "car": 4.6,
    "bus": 1.2,
    "train": 0.8,
    "bike": 0
}

def calculate_emissions(transport, miles):
    if transport not in transport_emissions:
        return "Transport type not recognized"

    emission_rate = transport_emissions[transport]
    total = emission_rate * miles
    return total

transport_type = "car"
distance = 15

emissions = calculate_emissions(transport_type, distance)

print("Transport Type:", transport_type)
print("Distance Traveled:", distance, "miles")
print("Estimated CO2 Emissions:", emissions, "kg")
