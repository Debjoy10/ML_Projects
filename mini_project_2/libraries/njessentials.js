function matrixmultiplication(M1, M2){
    var A = nj.array(M1);
    var B = nj.array(M2);
    var C = nj.dot(A, B);
    return C.tolist();
}

function matrixaddition(M1, M2){
    var A = nj.array(M1);
    var B = nj.array(M2);
    var C = A.add(B);
    return C.tolist();
}
