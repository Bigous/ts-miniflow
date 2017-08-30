import * as bm from "bluemath";
export type numbers = number | number[] | number[][] | null;

function matmul(mat: number[][], value: numbers): numbers {
  const ret = [];
  if (mat === null || value === null) {
    return null;
  }
  if (typeof value === "number") {
    for (const row of mat) {
      const line = [];
      for (const item of row) {
        line.push(item * value);
      }
      ret.push(line);
    }
  } else if (Array.isArray(value)) {
    let v: number[][] = value as number[][];
    if (!Array.isArray(value[0])) {
      v = [value] as number[][];
    }
    if (mat[0].length !== v.length) {
      throw new Error(
        `Its not possible to multiply this matrix: (${mat.length} x ${mat[0]
          .length}) . (${v.length} x ${v[0].length})`
      );
    }
    // tslint:disable-next-line:prefer-for-of
    for (let i = 0; i < mat.length; i++) {
      const line = [];
      for (let j = 0; j < v[0].length; j++) {
        let s = 0;
        for (let o = 0; o < v.length; o++) {
          s += mat[i][o] * v[o][j];
        }
        line.push(s);
      }
      ret.push(line);
    }
  }
  return ret;
}

export function dot(...value: numbers[]): numbers {
  let ret: numbers = null;
  for (const n of value) {
    if (ret === null) {
      ret = n;
    } else {
      if (Array.isArray(ret) && Array.isArray(ret[0])) {
        ret = matmul(ret as number[][], n);
      } else if (Array.isArray(n) && Array.isArray(n[0])) {
        ret = matmul(n as number[][], ret);
      } else {
        // tslint:disable-next-line:prefer-conditional-expression
        if (Array.isArray(ret)) {
          ret = matmul([ret] as number[][], n);
        } else if (Array.isArray(n)) {
          ret = matmul([n] as number[][], ret);
        } else {
          ret = n === null ? null : ret * n;
        }
      }
    }
  }
  return ret;
}

export function hat(mat: number[][]): number[][] {
  const ret = [];
  for (let i = 0; i < mat.length; i++) {
    const line = [];
    for (let o = 0; o < mat[0].length; o++) {
      line.push(mat[o][i]);
    }
    ret.push(line);
  }
  return ret;
}

export function inv(mat: bm.NDArray): bm.NDArray {
  const ret = new bm.NDArray({
    datatype: mat.datatype,
    shape: [mat.shape[1], mat.shape[0]]
  });
  for (let i = 0; i < mat.shape[0]; i++) {
    for (let o = 0; o < mat.shape[1]; o++) {
      ret.set(o, i, mat.get(i, o));
    }
  }
  return ret;
}

export function sum(
  mat: number[][],
  axis: null | number = null
): number | number[] {
  let ret;
  if (axis === null) {
    ret = 0;
    for (const l of mat) {
      for (const e of l) {
        ret += e;
      }
    }
  } else if (axis === 0) {
    // sum in columns
    ret = [];
    for (let i = 0; i < mat[0].length; i++) {
      let s = 0;
      // tslint:disable-next-line:prefer-for-of
      for (let o = 0; o < mat.length; o++) {
        s += mat[o][i];
      }
      ret.push(s);
    }
  } else {
    // sum in columns
    ret = [];
    // tslint:disable-next-line:prefer-for-of
    for (let i = 0; i < mat.length; i++) {
      let s = 0;
      for (let o = 0; o < mat[0].length; o++) {
        s += mat[i][o];
      }
      ret.push(s);
    }
  }
  return ret;
}

function addNumArray(x: number, y: number[]): number[] {
  return [...y].map(v => v + x);
}

function add2(x: numbers, y: numbers): numbers {
  if (x == null || y == null) {
    return null;
  }
  if (typeof x === "number" && typeof y === "number") {
    return x + y;
  }
  if (typeof x === "number") {
    if (Array.isArray(y)) {
      if (Array.isArray(y[0])) {
        const ret = [];
        for (const l of y as number[][]) {
          ret.push(addNumArray(x, l));
        }
        return ret;
      } else {
        return addNumArray(x, y as number[]);
      }
    }
  } else if (Array.isArray(x)) {
    if (typeof y === "number") {
      if (Array.isArray(x[0])) {
        const ret = [];
        for (const l of x as number[][]) {
          ret.push(addNumArray(y, l));
        }
        return ret;
      } else {
        return addNumArray(y, x as number[]);
      }
    } else {
      // ambos sao array...
      if (Array.isArray(x[0]) && Array.isArray(y[0])) {
        // ambos são arrays duplos - então tem que ter o mesmo "shape" e daí soma item a item de cada.
        const xaa = x as number[][];
        const yaa = y as number[][];
        if (xaa.length !== yaa.length || xaa[0].length !== yaa[0].length) {
          throw Error(`Shape mismatch adding X(${xaa.length},${xaa[0].length}) to Y(${yaa.length},${yaa[0].length})`);
        }
        const ret = [];
        for (let i = 0; i < xaa.length; i++) {
          const linha = [];
          for (let o = 0; o < xaa[0].length; o++) {
            linha.push(xaa[i][o] + yaa[i][o]);
          }
          ret.push(linha);
        }
        return ret;
      } else if (Array.isArray(x[0]) || Array.isArray(y[0])) {
        // pelo menos 1 é array duplo
        const xaa = (Array.isArray(x[0]) ? x : y) as number[][];
        const ya = (Array.isArray(x[0]) ? y : x) as number[];
        // TODO: continuar...
      }
    }
  }

  throw Error(`Cannot add (${typeof x}) ${x} to (${typeof y}) ${y}`);
}

export function add(...values: numbers[]): numbers {
  let ret: numbers = null;
  for (const value of values) {
    ret = ret === null ? value : add2(ret, value);
  }
  return ret;
}

export function sumColumns(mat: bm.NDArray): bm.NDArray {
  const ret = new bm.NDArray({
    datatype: mat.datatype,
    shape: [1, mat.shape[1]]
  });
  for (let col = 0; col < mat.shape[1]; col++) {
    let s = 0;
    for (let linha = 0; linha < mat.shape[0]; linha++) {
      s += mat.get(linha, col) as number;
    }
    ret.set(0, col, s);
  }
  return ret;
}

export function sigmoid(x: number): number {
  return 1 / (1 - Math.exp(-x));
}
