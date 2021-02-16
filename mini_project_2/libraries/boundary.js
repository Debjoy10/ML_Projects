// Inspied by the code from Daniel Shiffman on 2D Ray-Casting

class Boundary {
  constructor(x1, y1, x2, y2) {
    this.a = createVector(x1, y1);
    this.b = createVector(x2, y2);
  }

  lineline(x1,y1, x2, y2, x3, y3, x4, y4){
    let den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if(den==0)
        return false;
    else{
        let t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den;
        let u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den;
        if (t >= 0 && t <= 1 && u >= 0 && u<=1) {
            return true;
        } 
    }
    return false;
  }

  collision(x1, y1, x2, y2, x3, y3, x4, y4){
    if( this.lineline(this.a.x, this.a.y, this.b.x, this.b.y, x1, y1, x2, y2) ||
        this.lineline(this.a.x, this.a.y, this.b.x, this.b.y, x2, y2, x3, y3) ||
        this.lineline(this.a.x, this.a.y, this.b.x, this.b.y, x3, y3, x4, y4) ||
        this.lineline(this.a.x, this.a.y, this.b.x, this.b.y, x4, y4, x1, y1)){
        return true;
    } 
    return false;
  }
}
