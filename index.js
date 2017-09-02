/*
MIT License

Copyright (c) 2017 james jung (teetotal@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

'use strict';
/**
 * Dataset for prediction
 */
class collaborativeFilteringTargetDataset {
    constructor() {
        this.users = [];
        this.items = [];
        this.predicted = [];
    }

    /**
     * 
     * @param {String} user
     * @param {String} item
     */
    add(user, item) {
        this.users.push(user);
        this.items.push(item);
    }

    clearPrediction() {
        delete this.predicted;
        this.predicted = [];
    }

    /**
     * Get Predicted Result
     * @param {Number} fixed
     * @param {String} sortKey  user | item | predicted
     * @param {String} sortType    desc(default) | asc
     */
    getTable(fixed, sortKey, sortType) {
        let arr = [];
        for (let n = 0; n < this.users.length; n++) {
            let predicted;
            if (this.predicted[n] == null) {
                predicted = null
            } else {
                predicted = (fixed >= 0 ? this.predicted[n].toFixed(fixed) : this.predicted[n])
            }
            arr.push({
                user: this.users[n]
                , item: this.items[n]
                , predicted: predicted
            });
        }

        if (sortKey) {
            sortType = sortType ? sortType : "desc";
            arr.sort((a, b) => {
                if (sortKey === 'predicted') {
                    if (sortType === "desc")
                        return b[sortKey] - a[sortKey];
                    else
                        return a[sortKey] - b[sortKey];
                }
                else {
                    if (sortType === "desc") {
                        if (a[sortKey] > b[sortKey]) return -1;
                        else return 1;
                    } else {
                        if (b[sortKey] > a[sortKey]) return -1;
                        else return 1;
                    }
                }
            });
        }

        return arr;
    }
}

class hashArray {
    constructor() {
        this.hash = {};
        this.array = [];
    }

    getIdx(key) {
        if (this.hash[key] == null) {
            this.array.push(key);
            this.hash[key] = this.array.length - 1;
        }

        return this.hash[key];
    }

    getKey(idx) {
        return this.array[idx];
    }

    import(p) {
        delete this.hash;
        this.hash = {};
        delete this.array;
        this.array = [];

        for (let n = 0; n < p.array.length; n++) {
            this.array.push(p.array[n]);
            this.hash[p.array[n]] = this.array.length - 1;
        }
    }
}
/**
 * Dataset for training
 */
class collaborativeFilteringDataset {
    constructor(dimension) {
        this.meta = {
            users: new hashArray(),
            items: new hashArray()
        };
        this.users = {};
        this.items = {};
        this.dimension = dimension ? dimension : 2;        
    }

    /**
     * 
     * @param {String} user
     * @param {String} item
     * @param {Number} rating
     */
    add(user, item, rating) {
        const userIdx = this.meta.users.getIdx(user);
        const itemIdx = this.meta.items.getIdx(item);

        if (this.users[userIdx] == null) {
            this.users[userIdx] = {
                theta: [],
                ratings: [],
                items: []
            };

            for (let n = 0; n < this.dimension; n++) {
                this.users[userIdx].theta.push(Math.random());
            }
        }

        if (this.items[itemIdx] == null) {
            this.items[itemIdx] = {
                x: [],
                users: [],
                ratings: []
            };

            for (let n = 0; n < this.dimension; n++) {
                this.items[itemIdx].x.push(Math.random());
            }
        }

        this.users[userIdx].ratings.push(rating);
        this.users[userIdx].items.push(itemIdx);

        this.items[itemIdx].ratings.push(rating);
        this.items[itemIdx].users.push(userIdx);
    }

    getUsers(users) {
        if (users == null)
            return this.users;

        let arr = [];
        for (let n in users) {
            arr.push(this.users[users[n]].theta);
        }
        return arr;
    }

    getItems(items) {
        if (items == null)
            return this.items;

        let arr = [];
        for (let n in items) {
            arr.push(this.items[items[n]].x);
        }
        return arr;
    }

    export() {
        return {
            theta: this.users,
            x: this.items,
            meta: this.meta
        };
    }

    import(p) {
        this.users = p.theta;
        this.items = p.x;
        this.meta.users.import(p.meta.users);
        this.meta.items.import(p.meta.items);
    }
}

class collaborativeFiltering {
    constructor() {
    }
    /**
     * inner product
     */
    innerProduct(A, B) {
        let sum = 0;
        for (let n = 0; n < A.length; n++) {
            sum += A[n] * B[n];
        }
        return sum;
    }
    /**
     * Gradient Decent
     * @param {any} ratings =    ex) [5,4,3]
     * @param {any} theta   =    ex) [.23, .2]
     * @param {any} x       =    ex) [[.23, .2], [.1,.2], [.5, .6]]
     */
    gradientDecent(ratings, theta, x, lambda) {
        let val = [];
        for (let i = 0; i < theta.length; i++) {
            let sum = 0;
            for (var n = 0; n < ratings.length; n++) {
                let predicted = this.innerProduct(theta, x[n]);
                sum += (predicted - ratings[n]) * x[n][i];
            }
            sum += theta[i] * lambda;
            val[i] = sum;
        }
        return val;
    }
    
    /**
     * update theta or x
     * @param {any} dest destination
     * @param {any} src source
     * @param {any} alpha alpha
     */
    update(dest, src, alpha) {
        for (let n = 0; n < src.length; n++) {
            dest[n] = dest[n] - alpha * src[n];
        }
    }

    /**
     * training
     * @param {any} dataset
     * @param {Number} iterations option
     * @param {Number} lambda option
     * @param {Number} alpha option
     */
    training(dataset, iterations, lambda, alpha) {
        iterations = iterations ? iterations : 500;
        lambda = lambda ? lambda : 0;
        alpha = alpha ? alpha : 0.01;


        let users = dataset.getUsers();
        let items = dataset.getItems();

        for (let n = 0; n < iterations; n++) {

            for (let user in users) {
                let x = dataset.getItems(users[user].items);
                this.update(
                    users[user].theta
                    , this.gradientDecent(users[user].ratings, users[user].theta, x, lambda)
                    , alpha
                );
            }

            for (let item in items) {
                let theta = dataset.getUsers(items[item].users);
                this.update(
                    items[item].x
                    , this.gradientDecent(items[item].ratings, items[item].x, theta, lambda)
                    , alpha
                );
            }
        }
        return dataset;
    }

    /**
     * predict target data
     * @param {any} dataset
     * @param {any} target
     */
    transform(dataset, target) {
        const users = target.users;
        const items = target.items;
        let ratings = target.predicted;

        for (let n = 0; n < users.length; n++) {
            const predict = this.innerProduct(
                dataset.users[dataset.meta.users.getIdx(users[n])].theta
                , dataset.items[dataset.meta.items.getIdx(items[n])].x
            );

            ratings.push(predict);
        }

    }

    /**
     * predict
     * @param {any} dataset
     * @param {String} user
     * @param {String} item
     */
    predict(dataset, user, item) {
        return this.innerProduct(dataset.users[dataset.meta.users.getIdx(user)].theta, dataset.items[dataset.meta.items.getIdx(item)].x);
    }
}

exports.createInstance = function () {
    return new collaborativeFiltering();
}

exports.createDataset = function (dimension) {
    return new collaborativeFilteringDataset(dimension);
}

exports.createTargetDataset = function () {
    return new collaborativeFilteringTargetDataset();
}
