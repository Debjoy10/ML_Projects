class RaceTrack{
  constructor(num=0){
    this.load_racetrack(num);
  }
  load_racetrack(num){
    getTrack(this, num);
    this.setDims();
    this.createBoundaries();
  }
  setDims(){
    for(let i=0; i<this.inner.length; i++){
      for(let j=0; j<this.inner[i].length; j++){
        this.inner[i][j][0] *= width;
        this.inner[i][j][1] *= height;
        this.inner[i][j][0] += width*0.5;
        this.inner[i][j][1] += height*0.5;
      }
    }
    for(let i=0; i<this.outer.length; i++){
      this.outer[i][0] *= width;
      this.outer[i][1] *= height;
      this.outer[i][0] += width*0.5;
      this.outer[i][1] += height*0.5;
    }
  }

  createBoundaries(){
      this.boundaries = [];

      for(let i=0; i<this.inner.length; i++){
        for(let j=0; j<this.inner[i].length;j++){
          let n = (j+1)%(this.inner[i].length);
          this.boundaries.push(new Boundary(this.inner[i][j][0], this.inner[i][j][1], this.inner[i][n][0], this.inner[i][n][1]));
        }
      }
      for(let i=0; i<this.outer.length; i++){
        let n = (i+1)%(this.outer.length);
        this.boundaries.push(new Boundary(this.outer[i][0], this.outer[i][1], this.outer[n][0], this.outer[n][1]));
      }
  }

  draw(){

    fill(0);
    noStroke();
    beginShape();
    
    for(let i=0; i<this.outer.length; i++){
      vertex(this.outer[i][0], this.outer[i][1]);
    }
    
    endShape(CLOSE);

    fill(255);
    noStroke();
    
    for(let i=0; i<this.inner.length; i++){
      beginShape();
      for(let j=0; j<this.inner[i].length; j++){
        vertex(this.inner[i][j][0], this.inner[i][j][1]);
      }
      endShape(CLOSE);
    }

    drawStartLine();
  }
}
