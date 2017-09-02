let p = require('../index.js');
let ds = p.createDataset();

ds.add('Alice', 'Love at last', 5);
ds.add('Alice', 'Remance forever', 5);
ds.add('Alice', 'Nonstop car chases', 0);
ds.add('Alice', 'Sword vs. karate', 0);

ds.add('Bob', 'Love at last', 5);
ds.add('Bob', 'Cute puppies of love', 4);
ds.add('Bob', 'Nonstop car chases', 0);
ds.add('Bob', 'Sword vs. karate', 0);

ds.add('Carol', 'Love at last', 0);
ds.add('Carol', 'Cute puppies of love', 0);
ds.add('Carol', 'Nonstop car chases', 5);
ds.add('Carol', 'Sword vs. karate', 5);

ds.add('Dave', 'Love at last', 0);
ds.add('Dave', 'Remance forever', 0);
ds.add('Dave', 'Nonstop car chases', 4);

const cf = p.createInstance();
const training = cf.training(ds);

// Export trained dataset
const dump = training.export();

// Save exported dataset
const fs = require('fs');
fs.writeFileSync('./dump.json', JSON.stringify(dump, null, '\t'));

let target = p.createTargetDataset();
//let r = cf.predict(training, 'Alice', 'Love at last');

target.add('Alice', 'Love at last');
target.add('Alice', 'Cute puppies of love');
target.add('Alice', 'Remance forever');
target.add('Alice', 'Nonstop car chases');
target.add('Alice', 'Sword vs. karate');

target.add('Bob', 'Love at last');
target.add('Bob', 'Cute puppies of love');
target.add('Bob', 'Remance forever');
target.add('Bob', 'Nonstop car chases');
target.add('Bob', 'Sword vs. karate');

target.add('Carol', 'Love at last');
target.add('Carol', 'Cute puppies of love');
target.add('Carol', 'Remance forever');
target.add('Carol', 'Nonstop car chases');
target.add('Carol', 'Sword vs. karate');

target.add('Dave', 'Love at last');
target.add('Dave', 'Cute puppies of love');
target.add('Dave', 'Remance forever');
target.add('Dave', 'Nonstop car chases');
target.add('Dave', 'Sword vs. karate');

cf.transform(training, target);
console.log("Trainig", target.getTable(0, 'user', 'desc'));

// Load exported dataset
const pImport = JSON.parse(fs.readFileSync('./dump.json').toString());
// Create new dataset instance
let exportedDataset = p.createDataset();
// Import dataset
exportedDataset.import(pImport);
// Predict 
target.clearPrediction();
console.log("Clear", target.getTable(0));
cf.transform(exportedDataset, target);
console.log("Import", target.getTable(0));

function printPredict(user, item) {
    console.log(user, item, cf.predict(training, user, item).toFixed(0));
}

printPredict('Alice', 'Cute puppies of love');
printPredict("Bob", 'Remance forever');
printPredict("Carol", 'Remance forever');
printPredict("Dave", 'Cute puppies of love');
printPredict("Dave", 'Sword vs. karate');
