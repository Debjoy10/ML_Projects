let constructMode = false;
const maxPower = 0.075;
const maxReverse = 0.0375;
const powerFactor = 0.001;
const reverseFactor = 0.0005;
const drag = 0.95;
const angularDrag = 0.95;
let turnSpeed = 0.002;
let px = -5 , py = -10;
let skipchecks = 5;

class Car{
    constructor(){
        this.init_attr();
        this.rays = [];
        for(let i=360; i>=180; i-=30){    
          this.rays.push(new Ray(new p5.Vector(this.x, this.y), i));
        }
        this.network = new Network();
        this.pointreached = -1;
        this.fitness = 0;
    }
    update(){
      if(!constructMode && !this.stop){
        if (this.isThrottling) {
            this.power += powerFactor;
        } else {
            this.power -= powerFactor;
        }
        if (this.isReversing) {
            this.reverse += reverseFactor;
        } else {
            this.reverse -= reverseFactor;
        }

        this.power = Math.max(0, Math.min(maxPower, this.power));
        this.reverse = Math.max(0, Math.min(maxReverse, this.reverse));

        const direction = this.power > this.reverse ? 1 : -1;

        if (this.isTurningLeft) {
            this.angularVelocity -= direction * turnSpeed;
        }
        if (this.isTurningRight) {
            this.angularVelocity += direction * turnSpeed;
        }

        this.xVelocity += Math.sin(this.angle) * (this.power - this.reverse);
        this.yVelocity += Math.cos(this.angle) * (this.power - this.reverse);

        this.x += this.xVelocity;
        this.y -= this.yVelocity;
        this.xVelocity *= drag;
        this.yVelocity *= drag;
        this.angle += this.angularVelocity;
        this.angularVelocity *= angularDrag;
      }
    }
    updateRays(){
      for(let ray of this.rays){ 
        ray.pos.x = this.x;
        ray.pos.y = this.y;
        ray.setAngle(ray.groundAngle + this.angle*180/PI);
      }
    }
    predict(D){
      return this.network.forward(D);
    }
    restart(){
        this.init_attr();
    }
    init_attr(){
        this.x = windowWidth * 0.155;
        this.y = windowHeight / 2;
        this.xVelocity = 0;
        this.yVelocity= 0;
        this.power= 0;
        this.reverse= 0;
        this.angle= 0;
        this.angularVelocity= 0;
        this.isThrottling= false;
        this.isReversing= false;
        this.stop = false;
        this.colour = color(255,0,0);
    }
    draw(){
      push();
      translate(this.x,this.y);
      rotate(this.angle);
      fill(this.colour);
      noStroke();
      rectMode(CENTER);
      rect(0,0,-px*2,-py*2);
      pop();
      this.evaluateFitness();
    }
    checkBoundaries(){

      let xc,ys,xs,yc;
      xc = px*cos(this.angle);
      ys = py*sin(this.angle);
      xs = px*sin(this.angle);
      yc = py*cos(this.angle);

      let x1 = this.x + xc  - ys;
      let y1 = this.y + xs + yc;
      let x2 = this.x  - xc  - ys;
      let y2 = this.y  - xs + yc ;
      let x3 = this.x  - xc + ys;
      let y3 = this.y  - xs  - yc ;
      let x4 = this.x + xc + ys;
      let y4 = this.y + xs - yc ;

      for(let boundary of racetrack.boundaries){
        if(boundary.collision(x1, y1, x2, y2, x3, y3, x4, y4)){
            this.stop = true;
        }
      }
    }

    getClosestPoint(){
      let dmin = Infinity;
      let imin = null;

      for(let i=0; i<checkpoints.length; i++){
        let d = dist(checkpoints[i][0], checkpoints[i][1], this.x, this.y);
        if(d < dmin){
          dmin = d;
          imin = i;
        }
      }
      return imin;
    }

    evaluateFitness(){
      let p = this.getClosestPoint();

      if(p > this.pointreached){
        if(p - this.pointreached <= skipchecks){
            this.fitness += p - this.pointreached;
            this.pointreached = p;
        }
      }
      else if(p < this.pointreached){
        let differ = checkpoints.length - this.pointreached + p;
        if(differ <= skipchecks){
            this.fitness += differ;
            this.pointreached = p;
        }
      }
    }
}


let windowWidth = window.innerWidth;
let windowHeight = window.innerHeight;

function drawStartLine(){
    stroke(0,0,255);
    strokeWeight(10);
    line(width*0.5-width*0.4, height*0.5, width*0.5-width*0.3, height*0.5);
    strokeWeight(1);
}

function updateWorld(labels){
    if(!constructMode){
      for(j=0; j<evolution.pop.length; j++){
        let car = evolution.pop[j];
        let label = labels[j]; 

        const canTurn = car.power > 0.0025 || car.reverse;
        let pressingUp, pressingDown;

        pressingUp = boolean(label[0]);
        pressingDown = boolean(label[1]);

        if (car.isThrottling !== pressingUp || car.isReversing !== pressingDown) {
            car.isThrottling = pressingUp;
            car.isReversing = pressingDown;
        }

        let LEFT,RIGHT;

        LEFT = boolean(label[2]);
        RIGHT = boolean(label[3]);

        const turnLeft = canTurn && LEFT;
        const turnRight = canTurn && RIGHT;

        if (car.isTurningLeft !== turnLeft) {

            car.isTurningLeft = turnLeft;
        }
        if (car.isTurningRight !== turnRight) {

            car.isTurningRight = turnRight;
        }

        if (car.x > windowWidth) {
            car.x -= windowWidth;

        } else if (car.x < 0) {
            car.x += windowWidth;
        }

        if (car.y > windowHeight) {
            car.y -= windowHeight;

        } else if (car.y < 0) {
            car.y += windowHeight;
        }
      }
    }
}

let racetrack;
let evolution;
let checkpoints;

function setCheckpoints(){
   let checkpointsRel = [];
   if(num == 0){
        checkpointsRel = [[-0.35156,-0.04961],[-0.35352,-0.14883],[-0.35417,-0.25849],[-0.3112,-0.33812],[-0.23307,-0.35248],[-0.13802,-0.36815],[-0.05599,-0.37728],[0.02995,-0.37728],[0.09961,-0.36815],[0.17708,-0.3577],[0.2487,-0.35379],[0.32487,-0.33159],[0.35026,-0.23238],[0.34896,-0.1201],[0.34701,0.01044],[0.34635,0.11749],[0.34635,0.23629],[0.30339,0.32507],[0.23633,0.28982],[0.16146,0.26632],[0.08854,0.2389],[0.01237,0.21932],[-0.06185,0.25587],[-0.1263,0.28068],[-0.19401,0.30026],[-0.26172,0.31462],[-0.33073,0.31593],[-0.35417,0.21671],[-0.35742,0.10313]];
    }
    else if(num==1){
        checkpointsRel = [[-0.34635,-0.00261],[-0.34701,-0.05614],[-0.34766,-0.11488],[-0.34375,-0.16971],[-0.33919,-0.22324],[-0.34701,-0.29112],[-0.33854,-0.34726],[-0.3151,-0.39817],[-0.28125,-0.39556],[-0.25651,-0.38512],[-0.21615,-0.37206],[-0.17839,-0.36031],[-0.14388,-0.35248],[-0.10938,-0.34987],[-0.07161,-0.33812],[-0.03711,-0.32637],[-0.00195,-0.32637],[0.03516,-0.32637],[0.06445,-0.32115],[0.0944,-0.31593],[0.1276,-0.3107],[0.1582,-0.30809],[0.19271,-0.3094],[0.22135,-0.3094],[0.25456,-0.31332],[0.29167,-0.32637],[0.32617,-0.30548],[0.35286,-0.26501],[0.36914,-0.20888],[0.3763,-0.14883],[0.38542,-0.06919],[0.39583,-0.00783],[0.38151,0.05614],[0.36719,0.12272],[0.35352,0.17755],[0.31706,0.21279],[0.26497,0.22193],[0.22591,0.22715],[0.18099,0.2376],[0.14583,0.2389],[0.10417,0.25457],[0.06055,0.27285],[0.01823,0.27023],[-0.02669,0.28198],[-0.07747,0.28721],[-0.1224,0.29634],[-0.16992,0.30418],[-0.21354,0.3107],[-0.26042,0.31854],[-0.30078,0.32115],[-0.33073,0.3107],[-0.34375,0.23629],[-0.34831,0.17755],[-0.35026,0.07833]];
    }
    else if(num==2){
        checkpointsRel = [[-0.35091,-0.00653],[-0.35091,-0.03525],[-0.35091,-0.05614],[-0.34896,-0.07572],[-0.34245,-0.09661],[-0.33724,-0.11619],[-0.32943,-0.13969],[-0.32031,-0.15927],[-0.3099,-0.17102],[-0.30013,-0.18016],[-0.28906,-0.19191],[-0.2793,-0.19582],[-0.27148,-0.19843],[-0.26172,-0.20104],[-0.25326,-0.20104],[-0.24414,-0.20104],[-0.23372,-0.20104],[-0.22656,-0.21149],[-0.21615,-0.21932],[-0.20703,-0.22715],[-0.19922,-0.23238],[-0.1888,-0.24151],[-0.17643,-0.24413],[-0.16927,-0.25326],[-0.15625,-0.25326],[-0.14583,-0.25979],[-0.13477,-0.26371],[-0.1263,-0.26371],[-0.11523,-0.27154],[-0.10482,-0.27807],[-0.0957,-0.29243],[-0.08659,-0.29634],[-0.07682,-0.29896],[-0.06315,-0.3107],[-0.04948,-0.32115],[-0.03906,-0.32245],[-0.03255,-0.32245],[-0.02539,-0.32245],[-0.01563,-0.32245],[-0.00716,-0.32245],[0.00521,-0.32245],[0.01172,-0.31201],[0.02083,-0.30287],[0.0293,-0.29373],[0.03906,-0.28721],[0.05208,-0.27807],[0.0612,-0.27154],[0.06901,-0.2624],[0.07813,-0.25718],[0.08203,-0.25065],[0.08984,-0.24543],[0.09505,-0.24282],[0.10482,-0.2376],[0.11523,-0.23629],[0.12435,-0.23629],[0.13672,-0.23238],[0.14323,-0.23238],[0.15234,-0.23107],[0.16146,-0.23107],[0.17122,-0.22846],[0.18294,-0.22846],[0.19141,-0.22846],[0.19727,-0.22846],[0.20573,-0.23629],[0.21354,-0.2376],[0.22461,-0.2376],[0.22917,-0.2376],[0.23763,-0.23499],[0.24284,-0.22846],[0.25195,-0.22454],[0.25716,-0.22585],[0.26237,-0.22585],[0.27344,-0.22193],[0.27734,-0.20496],[0.28646,-0.18016],[0.28776,-0.16449],[0.29036,-0.14883],[0.29167,-0.12924],[0.29167,-0.10705],[0.29167,-0.08616],[0.29167,-0.05875],[0.29167,-0.04308],[0.29167,-0.01697],[0.29167,0.00653],[0.29167,0.02872],[0.28906,0.05483],[0.28581,0.08094],[0.2832,0.10705],[0.27865,0.12402],[0.27344,0.14752],[0.26823,0.1658],[0.26107,0.18277],[0.25195,0.20235],[0.24479,0.22063],[0.23242,0.23368],[0.22396,0.25849],[0.21289,0.25979],[0.20052,0.28198],[0.19076,0.29634],[0.18229,0.30026],[0.17253,0.31332],[0.16406,0.32245],[0.15104,0.32898],[0.14323,0.32898],[0.13151,0.32898],[0.11914,0.33159],[0.10677,0.33681],[0.0944,0.33681],[0.07813,0.33681],[0.0651,0.32507],[0.05273,0.31593],[0.04362,0.30809],[0.0319,0.29634],[0.02018,0.28851],[0.00846,0.27415],[-0.00195,0.26371],[-0.01237,0.24935],[-0.02344,0.23499],[-0.03646,0.22846],[-0.04688,0.21149],[-0.0612,0.20104],[-0.07031,0.19843],[-0.08268,0.19321],[-0.09245,0.18016],[-0.10417,0.17102],[-0.11458,0.17102],[-0.13151,0.17232],[-0.14128,0.17232],[-0.15104,0.17232],[-0.15625,0.17232],[-0.16797,0.16971],[-0.17969,0.16971],[-0.1888,0.16971],[-0.20508,0.16971],[-0.21484,0.1658],[-0.22656,0.15535],[-0.23763,0.1436],[-0.25065,0.13577],[-0.25781,0.13577],[-0.26823,0.13055],[-0.2806,0.12924],[-0.29297,0.12402],[-0.30469,0.1188],[-0.3151,0.11227],[-0.32487,0.11227],[-0.33529,0.11227],[-0.3431,0.09791],[-0.34831,0.07311],[-0.35091,0.05614]];
    }
    else if(num==3){
        checkpointsRel = [[-0.34961,-0.00783],[-0.32422,-0.06919],[-0.29818,-0.11488],[-0.26758,-0.15535],[-0.23568,-0.18407],[-0.20703,-0.22063],[-0.17188,-0.22324],[-0.13086,-0.25979],[-0.09766,-0.26632],[-0.06771,-0.27285],[-0.02604,-0.27937],[0.01172,-0.28982],[0.04883,-0.28198],[0.08203,-0.27937],[0.10417,-0.2624],[0.14063,-0.25587],[0.17253,-0.25196],[0.19792,-0.24674],[0.24154,-0.24282],[0.26042,-0.22715],[0.28125,-0.20104],[0.29362,-0.16319],[0.3099,-0.10313],[0.32227,-0.06527],[0.32227,-0.01436],[0.3099,0.047],[0.26367,0.07572],[0.22852,0.08355],[0.19531,0.09791],[0.15365,0.11619],[0.11914,0.13838],[0.08854,0.1671],[0.07422,0.19191],[0.04427,0.23238],[0.03385,0.2846],[0.00195,0.30548],[-0.03841,0.27937],[-0.08073,0.25718],[-0.11458,0.24935],[-0.15104,0.25196],[-0.18685,0.19974],[-0.21159,0.16841],[-0.23763,0.11488],[-0.27083,0.11358],[-0.3151,0.10705],[-0.33594,0.07963]];
    }
    else if(num==4){
        checkpointsRel =[[-0.34961,-0.00522],[-0.35026,-0.05483],[-0.35091,-0.09791],[-0.35091,-0.14491],[-0.34635,-0.20627],[-0.33659,-0.24282],[-0.31706,-0.26501],[-0.28971,-0.27415],[-0.26172,-0.27676],[-0.23828,-0.27546],[-0.21094,-0.27937],[-0.1862,-0.2859],[-0.16341,-0.2859],[-0.13607,-0.29112],[-0.11654,-0.28982],[-0.09115,-0.28982],[-0.07096,-0.28982],[-0.05013,-0.28851],[-0.02539,-0.28982],[-0.00586,-0.28982],[0.01367,-0.28198],[0.04167,-0.2846],[0.06315,-0.2846],[0.08333,-0.2846],[0.10547,-0.28851],[0.12565,-0.28982],[0.15104,-0.28982],[0.17448,-0.28982],[0.19401,-0.28982],[0.21484,-0.28982],[0.23828,-0.28982],[0.25391,-0.28982],[0.26563,-0.28198],[0.2806,-0.27676],[0.29948,-0.25718],[0.31445,-0.23238],[0.31836,-0.18799],[0.31901,-0.15927],[0.32096,-0.11619],[0.32096,-0.0705],[0.32096,-0.01567],[0.32096,0.02742],[0.32096,0.07441],[0.32096,0.11097],[0.32096,0.15274],[0.31771,0.20366],[0.31055,0.23238],[0.30143,0.25849],[0.28516,0.27415],[0.26302,0.27415],[0.24219,0.27807],[0.21875,0.27546],[0.20117,0.27154],[0.1849,0.26371],[0.17318,0.22193],[0.16406,0.1893],[0.15169,0.16057],[0.13607,0.1436],[0.11393,0.1201],[0.09375,0.10444],[0.06836,0.09008],[0.04688,0.08094],[0.02474,0.08225],[0.00846,0.08094],[0.00065,0.07963],[-0.01042,0.07572],[-0.01823,0.07572],[-0.02995,0.07572],[-0.03971,0.07311],[-0.05143,0.07311],[-0.06055,0.07311],[-0.06901,0.07963],[-0.08073,0.08225],[-0.08789,0.08747],[-0.10156,0.09269],[-0.11133,0.10183],[-0.11784,0.12663],[-0.12435,0.13969],[-0.13281,0.16449],[-0.13867,0.1893],[-0.14974,0.21932],[-0.16146,0.24282],[-0.17253,0.26632],[-0.18229,0.27546],[-0.19466,0.2846],[-0.21289,0.29896],[-0.22917,0.29896],[-0.24284,0.29896],[-0.25586,0.30287],[-0.27148,0.29896],[-0.28906,0.29896],[-0.30664,0.2846],[-0.31771,0.25849],[-0.33203,0.22063],[-0.3418,0.19191],[-0.34635,0.14752],[-0.34896,0.11358],[-0.34961,0.08616]];
    }
  checkpoints = [];
  for(let i=0; i<checkpointsRel.length; i++){
    let x,y;
    x = checkpointsRel[i][0]*width;
    y = checkpointsRel[i][1]*height;
    x += width*0.5;
    y += height*0.5;
    checkpoints.push([x,y]);
  }
}

let interval;

function setup(){
  createCanvas(windowWidth, windowHeight);
  if(!constructMode){
      background(0);

      setCheckpoints();
      racetrack = new RaceTrack(num);
      evolution = new Evolution();
      evolution.startLife();

      interval = setInterval(checkEvolution, 1500);
  }
  else{
    drawStartLine();
  }
}


function restart(){
  if(!constructMode){
  }
}

function checkEvolution(){
  if(!evolution.updateFitness()){
    evolution.select();
    evolution.mutateGeneration();
  }
}

function draw() {
    if(!constructMode){
        background(255);

        racetrack.draw();

        for(let i=0; i<10; i++){
            updateWorld(drawvisual());
            for(let car of evolution.pop){
              car.update();
              car.checkBoundaries();
            }
        }

        let maximum = -1;
        let maxcar = null;

        for(let car of evolution.pop){
            car.updateRays();

            for (let i = 0; i < car.rays.length; i++) {
                    const ray = car.rays[i];
                      let closest = null;
                      let record = Infinity;
                      for (let boundary of racetrack.boundaries) {
                        const pt = ray.cast(boundary);
                        if (pt) {
                          const d = dist(car.x, car.y, pt.x, pt.y);
                          if (d < record) {
                            record = d;
                            closest = pt;
                          }
                        }
                        }

                      if (closest) {
                        stroke(0,255,0,200);
                        strokeWeight(0.5);
                        line(car.x, car.y, closest.x, closest.y);
                      }
                    }



            car.draw();
            car.checkBoundaries();
            if(maximum < car.fitness){
                maximum = car.fitness;
                maxcar = car;
            }
            car.colour = color(255,0,0);
        }
        if(maxcar!=null){
          maxcar.colour = color(0,0,255);
          maxcar.draw();
      }

          // for(let p of fitpoints){
          //   fill(255);
          //   noStroke();
          //   ellipse(p[0], p[1], 10, 10);
          // }

        //   for(let p of checkpoints){
        //     fill(255);
        //   noStroke();
        //    ellipse(p[0], p[1], 10, 10);
        // }
        textSize(16);
        fill(0);
        stroke(0);
        text("Generation: "+evolution.generation, 10, 30);
        text("Max fitness: "+evolution.maxfitvals[evolution.maxfitvals.length-1], 10, 60);
    }
}

function drawvisual(){
  if(!constructMode){
    labels = [];
    let data = [0,0,0,0,0,0,0];
    for(let car of evolution.pop){ 
      for (let i = 0; i < car.rays.length; i++) {
          const ray = car.rays[i];
          let closest = null;
          let record = Infinity;
          for (let boundary of racetrack.boundaries) {
            const pt = ray.cast(boundary);
            if (pt) {
              const d = dist(car.x, car.y, pt.x, pt.y);
              if (d < record) {
                record = d;
                closest = pt;
              }
            }
          }

          data[i] = record;

          // if (closest) {
          //   stroke(0,255,0,200);
          //   strokeWeight(0.5);
          //   line(car.x, car.y, closest.x, closest.y);
          // }
        }
        labels.push(car.predict(data));
      }
      return labels;
  }
}
