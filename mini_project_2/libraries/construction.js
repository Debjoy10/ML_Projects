let construction_equipment = {
  outerpoly: [],
  innerpoly: [],
  outer: [],
  inner: [],
  index: -1
};

function fixFloatPrec(f,p){
  return float(f.toFixed(p));
}

function convXtoRelScale(x){
    return fixFloatPrec(-0.5 + x/width,5);
}
function convYtoRelScale(y){
    return fixFloatPrec(-0.5 + y/height,5);
}
function convVectoRelScale(vec){
    return [convXtoRelScale(vec.x), convYtoRelScale(vec.y)];
}

function convPolyToRelScale(poly){
        let res = [];

        for(let i=0; i<poly.length; i++){
            res.push(convVectoRelScale(poly[i]));
        }
        return res;
}

function convToRelScale(){

    construction_equipment.outer = convPolyToRelScale(construction_equipment.outerpoly);
    construction_equipment.inner = [];

    for(let i=0; i<construction_equipment.innerpoly.length; i++){
        construction_equipment.inner.push(convPolyToRelScale(construction_equipment.innerpoly[i]));    
    }
}

function keyPressed(){
  if(constructMode){
    if(keyCode == 32){
      if(construction_equipment.index!=-2)
        construction_equipment.index += 1;
    }
    else if(keyCode == 81){
        construction_equipment.index = -2;
        convToRelScale();
        console.log(
            "obj.outer = "+
            JSON.stringify(construction_equipment.outer)
            + ";\n\nobj.inner = "+
            JSON.stringify(construction_equipment.inner)
            + ";\n"
            );
    }
  }
  // else{
  //   if(keyCode==81){
  //     let res = [];
  //     for(let p of fitpoints){
  //       res.push(
  //         [
  //         convXtoRelScale(p[0]), 
  //         convYtoRelScale(p[1])
  //         ]
  //         );
  //     }
  //     console.log(JSON.stringify(res));
  //   }
  // }
}

let fitpoints = [];

function mousePressed(){

  function dispPoly(fore, poly){
      fill(fore);
      noStroke();
      beginShape();
      for(let i=0; i<poly.length; i++){
        vertex(poly[i].x, poly[i].y);
      }
      endShape();
  }

  if(constructMode){
    
    if(construction_equipment.index == -1){
      construction_equipment.outerpoly.push(createVector(mouseX, mouseY));
      background(255);
      dispPoly(0,construction_equipment.outerpoly);
    }
    else if(construction_equipment.index >=0){
      if(construction_equipment.innerpoly.length==construction_equipment.index)
        construction_equipment.innerpoly.push([]);
      construction_equipment.innerpoly[construction_equipment.index].push(createVector(mouseX, mouseY));
      background(255);
      dispPoly(0,construction_equipment.outerpoly);
      for(let i=0; i<construction_equipment.innerpoly.length; i++)
        dispPoly(255,construction_equipment.innerpoly[i]);
    }

    drawStartLine();
  }
  // else{
  //   fitpoints.push([mouseX, mouseY]);
  // }
}
