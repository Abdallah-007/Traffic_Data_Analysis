db.chicago_crashes.aggregate([
    {
      $project: {
        crash_date: { $dateFromString: { dateString: "$CRASH_DATE" } },
        day_of_week: { $dayOfWeek: { $dateFromString: { dateString: "$CRASH_DATE" } } },
        hour: { $hour: { $dateFromString: { dateString: "$CRASH_DATE" } } },
        weather: "$WEATHER_CONDITION",
        lighting: "$LIGHTING_CONDITION",
        road_condition: "$ROADWAY_SURFACE_COND",
        primary_cause: "$PRIMARY_CONTRIBUTORY_CAUSE",
        injuries: { $ifNull: ["$INJURIES_TOTAL", 0] },
        fatal: { $cond: [{ $gt: [{ $ifNull: ["$INJURIES_FATAL", 0] }, 0] }, 1, 0] },
        latitude: { $toDouble: "$LATITUDE" },
        longitude: { $toDouble: "$LONGITUDE" }
      }
    },
    { $match: { latitude: { $ne: null }, longitude: { $ne: null } } },
    { $out: "crashes_ml_ready" }
  ]);