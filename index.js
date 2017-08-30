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

    add(user, item) {
        this.users.push(user);
        this.items.push(item);
    }

    getTable(fixed) {
        let arr = [];
        for (let n = 0; n < this.users.length; n++) {
            arr.push({
                user: this.users[n]
                , item: this.items[n]
                , predicted: (fixed >= 0 ? this.predicted[n].toFixed(fixed) : this.predicted[n])
            });
        }

        return arr;
    }
}
/**
 * Dataset for training
 */
class collaborativeFilteringDataset {
    constructor(dimension) {
        this.users = {};
        this.items = {};
        this.dimension = dimension ? dimension : 2;
        this.isCleanup = false;
    }

    add(user, item, rating) {
        if (this.users[user] == null) {
            this.users[user] = {
                theta: [],
                ratings: [],
                items: []
            };

            for (let n = 0; n < this.dimension; n++) {
                this.users[user].theta.push(Math.random());
            }
        }

        this.users[user].ratings.push(rating);
        this.users[user].items.push(item);

        if (this.items[item] == null) {
            this.items[item] = {
                x: [],
                users: [],
                ratings: []
            };

            for (let n = 0; n < this.dimension; n++) {
                this.items[item].x.push(Math.random());
            }
        }

        this.items[item].ratings.push(rating);
        this.items[item].users.push(user);
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

    cleanup() {
        if (this.isCleanup === false) {
            for (let user in this.users) {
                delete this.users[user].ratings;
                delete this.users[user].items;
            }

            for (let item in this.items) {
                delete this.items[item].ratings;
                delete this.items[item].users;
            }
            this.isCleanup = true;
        }
    }

    export() {
        this.cleanup();
        return {
            theta: this.users,
            x: this.items
        };
    }

    import(p) {
        this.users = p.theta;
        this.items = p.x;
        this.isCleanup = true;

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
     * assign random value
     * @param {any} length
     * @param {any} dimension
     */
    setRandom(length, dimension) {
        let val = [];
        for (let n = 0; n < length; n++) {
            let ele = [];
            for (let m = 0; m < (dimension ? dimension : this.dimension); m++) {
                ele[m] = Math.random();
            }
            val[n] = ele;
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
     * @param {any} iterations
     * @param {any} lambda
     * @param {any} alpha
     */
    fit(dataset, iterations, lambda, alpha) {
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
        dataset.cleanup();
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
                dataset.users[users[n]].theta
                , dataset.items[items[n]].x
            );

            ratings.push(predict);
        }

    }

    /**
     * predict
     * @param {any} dataset
     * @param {any} user
     * @param {any} item
     */
    predict(dataset, user, item) {
        return this.innerProduct(dataset.users[user].theta, dataset.items[item].x);
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
