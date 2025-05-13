// Create a file named "mapReduceWeatherAnalysis.js":

var mapFunction = function() {
    if (this.WEATHER_CONDITION) {
        emit(this.WEATHER_CONDITION, {
            count: 1,
            injuries: this.INJURIES_TOTAL || 0,
            fatal: (this.INJURIES_FATAL > 0) ? 1 : 0
        });
    }
};

var reduceFunction = function(key, values) {
    var result = { count: 0, injuries: 0, fatal: 0 };
    
    values.forEach(function(value) {
        result.count += value.count;
        result.injuries += value.injuries;
        result.fatal += value.fatal;
    });
    
    return result;
};

var finalizeFunction = function(key, reducedValue) {
    reducedValue.injuryRate = reducedValue.injuries / reducedValue.count;
    reducedValue.fatalityRate = reducedValue.fatal / reducedValue.count;
    return reducedValue;
};

db.chicago_crashes.mapReduce(
    mapFunction,
    reduceFunction,
    {
        out: "crashes_by_weather",
        finalize: finalizeFunction
    }
);