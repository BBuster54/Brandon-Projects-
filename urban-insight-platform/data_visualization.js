// Sample dataset representing city environmental metrics
const cityData = [
  { city: "New York", airQuality: 65, trafficIndex: 78 },
  { city: "Los Angeles", airQuality: 58, trafficIndex: 82 },
  { city: "Chicago", airQuality: 70, trafficIndex: 64 },
  { city: "Boston", airQuality: 72, trafficIndex: 59 }
];

function displayCityData() {
  console.log("Urban Environmental Metrics:");

  cityData.forEach(city => {
    console.log(
      city.city +
      " | Air Quality Index: " +
      city.airQuality +
      " | Traffic Index: " +
      city.trafficIndex
    );
  });
}

displayCityData();
