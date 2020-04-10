var db = connect("localhost:27017/yelp");
var categories = db.categories
var business = db.business
var review = db.review

var convertString2ListCategories = function () {
    /**
     * Convert a comma separated string with categories into list with categories.
     */
    var bulkOp = business.initializeUnorderedBulkOp();

    var category2alias = function (x) {
        return categories.find({ "title": { "$in": x } }, { "alias": 1, "_id": 0 }).map(x => x.alias);
    };

    // CONVERT THE COMMA SEPARATED STRING OF CATEGORIES INTO AN ARRAY OF CATEGORIES
    var count = 0;
    var total_number = business.count();
    business.find({}).forEach(function (doc) {
        try {
            if (typeof doc.categories != "string")
                return;
            c = doc.categories.split(",").map(x => x.trim());
            bulkOp.find({ '_id': doc._id }).updateOne({
                '$set': { 'categories': c, 'categories_alias': category2alias(c) }
            });
            count++;

        } catch (error) {
            printjson(error);
        }
        if (count % 1000 === 0) {
            bulkOp.execute();
            bulkOp = business.initializeUnorderedBulkOp();
            printjson("Updated: " + count + "/" + total_number);
        }
    });

    // Clean up queues
    if (count > 0) {
        bulkOp.execute();
        printjson("Updated: " + count + "/" + total_number);
    }

};

// convertString2ListCategories();

// convert the string date to date
dateConversionStage = {
    $set: {
        convertedDate: { $toDate: "$date" }
    }
};

printjson(review.updateMany({ "convertedDate": { '$exists': 0 } }, [dateConversionStage]))


// add timestamp to each document in review where the timestamp is the difference in days between 12.10.2004 and the date of each reviwe

var minDate = ISODate("2016-01-01T00:00:00.000Z");

printjson(review.updateMany({}, [{ $addFields: { "timestamp": { $divide: [{ $subtract: ["$convertedDate", minDate] }, 1000 * 60 * 60 * 24] } } }]))



printjson(review.aggregate([
    { $group: { _id: {}, "minDate": { $min: "$convertedDate" }, "maxDate": { $max: "$convertedDate" } } },
    { $project: { "minDate": 1, "maxDate": 1, "difference": { $subtract: ["$maxDate", "$minDate"] } } }

]).toArray());

var d = ["shopping"]
d.forEach(cat=>{
    r = review.aggregate([
        { $match: { "convertedDate": { $gte: minDate } } },
        {
            $lookup: {
                from: "business",
                localField: "business_id",
                foreignField: "business_id",
                as: "business"
            }
        },
        { $unwind: { "path": "$business", "preserveNullAndEmptyArrays": false } },
        { $match: { "business.categories_alias": cat } },
        { $addFields: { "timestamp": { $divide: [{ $subtract: ["$convertedDate", minDate] }, 1000 * 60 * 60 * 24] } } },
        { $project: { "text": 1, "convertedDate": 1, "timestamp": 1, "user_id": 1, "business_id": 1 } },
        { $out: "review_" + cat }
    ]);
});