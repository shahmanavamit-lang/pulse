const assert=require('node:assert/strict');
const model=require('../dist/model.json');
const fixtures=require('./parity-fixtures.json');
const {predict,validate}=require('../dist/predictor.js');
for(const row of fixtures){
 const result=predict(model,row.inputs);
 assert.ok(Math.abs(result.score-row.probability)<1e-12,'Browser export must match scikit-learn');
 assert.ok(Math.abs(result.logit-(result.baseline+result.contributions.reduce((s,c)=>s+c.value,0)))<1e-10,'Contributions must reconstruct model logit');
 assert.equal(result.positive,row.probability>=.5);
}
const input=fixtures[0].inputs;
for(const k of [...model.numeric,...model.categorical]){
 assert.ok(validate(model,{...input,[k]:''}).length);
 assert.ok(validate(model,{...input,[k]:'garbage'}).length);
}
assert.throws(()=>predict(model,{...input,age:18}));
assert.throws(()=>predict(model,{...input,thal:0}));
assert.throws(()=>predict(model,{...input,age:50.5}));
assert.throws(()=>predict(model,{...input,chol:Infinity}));
assert.throws(()=>predict(model,{...input,oldpeak:-1}));
const split=require('../model/split.json');
assert.equal(new Set([...split.trainOriginalRowIds,...split.testOriginalRowIds]).size,297);
assert.equal(split.testOriginalRowIds.length,60);
console.log(`Passed ${fixtures.length} scikit-learn parity cases, exact contribution reconstruction, missing/invalid input checks, and disjoint train/test split.`);
