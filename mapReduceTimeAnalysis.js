var mapFunction = function() {
    var date = new Date(this.CRASH_DATE);
    var dayOfWeek = date.getDay();  // 0 = Sunday, 1 = Monday, etc.
    var hour = date.getHours();
    
    var dayNames = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    emit(dayNames[dayOfWeek] + "_" + hour, 1);
};

var reduceFunction = function(key, values) {
    return Array.sum(values);
};

db.chicago_crashes.mapReduce(
    mapFunction,
    reduceFunction,
    { out: "crashes_by_time" }
);