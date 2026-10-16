import { calculate, validate } from "./model.js";
import { el, format, colors } from "./render.js";
export const presets = {
  train: input => ({...input,transport:"train"}),
  vegetarian: input => ({...input,diet:"vegetarian"}),
  flights: input => ({...input,flights:0}),
  electricity: input => ({...input,electricity:Number(input.electricity)*.8})
};
export function comparison(baseline, scenario) {
  return baseline.categories.map(b=>{const next=scenario.categories.find(c=>c.id===b.id);const saving=b.value-next.value;return {...b,scenario:next.value,saving,percent:b.value ? saving/b.value*100 : null};});
}
export function initScenarios({getInput,setInput,update,data}) {
  let baseline=null, result=null, input=null;
  const body=document.querySelector("#comparison-body"), content=document.querySelector("#comparison-content"), note=document.querySelector("#scenario-note");
  const goal=document.querySelector("#goal");
  function refresh(){
    if(!result)return;
    if(baseline) {
      const before=calculate(baseline,data),delta=before.total-result.total;
      content.hidden=false;body.replaceChildren();
      document.querySelector("#baseline-total").textContent=format(before.total)+" kg";
      document.querySelector("#scenario-total").textContent=format(result.total)+" kg";
      document.querySelector("#scenario-delta").textContent=(delta>=0?"Saving: ":"Increase: ")+format(Math.abs(delta))+" kg"+(before.total?" ("+format(Math.abs(delta/before.total*100))+"%)":" (baseline is zero; percentage unavailable)");
      const rows=comparison(before,result),scale=Math.max(1,...rows.map(r=>Math.abs(r.saving))),bars=document.querySelector("#savings-bars");bars.replaceChildren();
      for(const r of rows){
        const row=el("tr");row.append(el("th",r.label),el("td",format(r.value)),el("td",format(r.scenario)),el("td",(r.saving>=0?"−":"+")+format(Math.abs(r.saving))),el("td",r.percent===null?"—":(r.percent>=0?"−":"+")+format(Math.abs(r.percent))+"%"));body.append(row);
        const line=el("div",undefined,"saving-row");line.append(el("span",r.label));const track=el("span",undefined,"saving-track"),bar=el("span");bar.style.width=(Math.abs(r.saving)/scale*50)+"%";bar.style.backgroundColor=colors[r.id];bar.className=r.saving<0?"increase":"saving";track.append(bar);line.append(track,el("span",(r.saving>=0?"−":"+")+format(Math.abs(r.saving))+" kg"));bars.append(line);
      }
    } else content.hidden=true;
    refreshGoal();
  }
  function refreshGoal(){
    const raw=goal.value,n=Number(raw),error=document.querySelector("#goal-error");
    if(raw==="" ){error.textContent="";goal.setAttribute("aria-invalid","false");document.querySelector("#goal-progress").hidden=true;return;}
    if(!Number.isFinite(n)||n<=0||n>10000000000){error.textContent="Enter a target greater than 0 and at most 10 billion kg.";goal.setAttribute("aria-invalid","true");document.querySelector("#goal-progress").hidden=true;return;}
    error.textContent="";goal.setAttribute("aria-invalid","false");if(!result)return;
    document.querySelector("#goal-progress").hidden=false;
    const progress=document.querySelector("#goal-meter");progress.max=n;progress.value=Math.min(n,result.total);
    document.querySelector("#goal-status").textContent=format(result.total/n*100)+"% of your "+format(n)+" kg monthly budget used. "+(result.total<=n?format(n-result.total)+" kg below target.":format(result.total-n)+" kg above target.");
  }
  document.querySelector("#capture-baseline").addEventListener("click",()=>{const next=getInput();if(!validate(next).valid)return;baseline={...next};note.textContent="Baseline captured. Apply a preset or edit Your month to build your scenario.";refresh();});
  document.querySelector("#reset-scenario").addEventListener("click",()=>{if(baseline)setInput(baseline);baseline=null;note.textContent="Comparison cleared. Your original inputs are restored.";update();});
  for(const button of document.querySelectorAll("[data-preset]"))button.addEventListener("click",()=>{
    const next=getInput();if(!validate(next).valid)return;if(!baseline)baseline={...next};
    setInput(presets[button.dataset.preset](next));note.textContent="Scenario applied. Presets build on your current inputs; reset to restore the baseline.";update();
  });
  goal.addEventListener("input",refreshGoal);
  document.addEventListener("footprint-invalid",()=>{document.querySelector("#capture-baseline").disabled=true;for(const b of document.querySelectorAll("[data-preset]"))b.disabled=true;});
  document.addEventListener("footprint-update",e=>{
    ({input,result}=e.detail);document.querySelector("#capture-baseline").disabled=false;for(const b of document.querySelectorAll("[data-preset]"))b.disabled=false;
    if(baseline&&(baseline.scope!==input.scope||Number(baseline.household)!==Number(input.household))){baseline=null;note.textContent="Comparison reset because household size or reporting scope changed. Capture a new baseline.";}
    refresh();
  });
  return {get baseline(){return baseline;}, restore(saved){if(saved&&validate(saved).valid)baseline={...saved};}};
}

