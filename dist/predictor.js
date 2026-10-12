(function(root,factory){if(typeof module==='object'&&module.exports)module.exports=factory();else root.PulsePredictor=factory();})(typeof globalThis!=='undefined'?globalThis:this,function(){
 'use strict';
 function validate(model,inputs){
   const errors=[];
   for(const key of [...model.numeric,...model.categorical]){
     const raw=inputs[key];
     if(raw===undefined||raw===null||String(raw).trim()===''){errors.push({key,reason:'missing'});continue;}
     const value=Number(raw);
     if(!Number.isFinite(value)){errors.push({key,reason:'invalid'});continue;}
     if(model.numeric.includes(key)){
       const [min,max]=model.ranges[key];
       if(value<min||value>max)errors.push({key,reason:'range',min,max});
       else if(key!=='oldpeak'&&!Number.isInteger(value))errors.push({key,reason:'integer'});
     }else if(!model.categories[model.categorical.indexOf(key)].includes(value))errors.push({key,reason:'category'});
   }return errors;
 }
 function predict(model,inputs){
   const errors=validate(model,inputs);if(errors.length)throw new Error('Invalid model inputs: '+errors.map(e=>e.key+': '+e.reason).join(', '));
   const encoded=model.numeric.map((k,i)=>(Number(inputs[k])-model.means[i])/model.scales[i]);
   model.categorical.forEach((k,i)=>model.categories[i].forEach(v=>encoded.push(Number(inputs[k])===v?1:0)));
   const logit=model.intercept+encoded.reduce((s,v,i)=>s+v*model.weights[i],0);
   const score=logit>=0?1/(1+Math.exp(-logit)):Math.exp(logit)/(1+Math.exp(logit));
   let offset=0;const contributions=[];
   model.numeric.forEach(key=>{const i=offset++;contributions.push({key,value:(encoded[i]-model.encodedTrainingMean[i])*model.weights[i]});});
   model.categorical.forEach((key,ci)=>{let value=0;for(let j=0;j<model.categories[ci].length;j++){const i=offset++;value+=(encoded[i]-model.encodedTrainingMean[i])*model.weights[i];}contributions.push({key,value});});
   const baseline=model.intercept+model.encodedTrainingMean.reduce((s,v,i)=>s+v*model.weights[i],0);
   return {score,positive:score>=model.threshold,contributions:contributions.sort((a,b)=>Math.abs(b.value)-Math.abs(a.value)),logit,baseline};
 }
 return {validate,predict};
});
