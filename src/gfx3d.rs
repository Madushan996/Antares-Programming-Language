//! 3D graphics: a camera, lights, shapes and models, drawn on the graphics
//! card into the same picture as 2D drawing, so labels and effects can go
//! on top.
//!
//! A program describes each frame as it draws it: it places the camera,
//! the sun and lamps, and the shapes, and `render` (or `frame`) draws them.
//! The look aims for realism with little effort: light is worked out the
//! physically based way (rough and shiny, plain and metal surfaces), the sun
//! casts soft shadows, the sky lights everything a little, distance can
//! fade into fog, and edges are smoothed (4x multisampling).
//!
//! The drawing itself is done by Direct3D 11 (`d3d` below), which every
//! Windows PC has. Without a graphics card, Windows' own software renderer
//! (WARP) is used instead, so 3D works everywhere on Windows; elsewhere
//! `render` says 3D is Windows-only for now.

use crate::builtins::{num, text, want};
use crate::interp::Interp;
use crate::value::Value;
use std::cell::RefCell;
use std::rc::Rc;

type Res = Result<Value, String>;
type V3 = [f32; 3];
/// A 4 by 4 matrix, row by row, for row vectors: p' = p M.
type M4 = [f32; 16];

// ---------- vectors and matrices ----------

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn norm(a: V3) -> V3 {
    let l = dot(a, a).sqrt();
    if l > 0.0 { [a[0] / l, a[1] / l, a[2] / l] } else { [0.0, 1.0, 0.0] }
}

fn mmul(a: &M4, b: &M4) -> M4 {
    let mut r = [0f32; 16];
    for i in 0..4 {
        for j in 0..4 {
            r[i * 4 + j] = (0..4).map(|k| a[i * 4 + k] * b[k * 4 + j]).sum();
        }
    }
    r
}

const IDENTITY: M4 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

fn translate(t: V3) -> M4 {
    let mut m = IDENTITY;
    m[12] = t[0];
    m[13] = t[1];
    m[14] = t[2];
    m
}

fn scaling(s: V3) -> M4 {
    let mut m = IDENTITY;
    m[0] = s[0];
    m[5] = s[1];
    m[10] = s[2];
    m
}

/// Turning by `deg` degrees about x, then y, then z (anticlockwise when
/// looking along the axis towards the middle).
fn turning(deg: V3) -> M4 {
    let (sx, cx) = deg[0].to_radians().sin_cos();
    let (sy, cy) = deg[1].to_radians().sin_cos();
    let (sz, cz) = deg[2].to_radians().sin_cos();
    let rx = [1.0, 0.0, 0.0, 0.0, 0.0, cx, sx, 0.0, 0.0, -sx, cx, 0.0, 0.0, 0.0, 0.0, 1.0];
    let ry = [cy, 0.0, -sy, 0.0, 0.0, 1.0, 0.0, 0.0, sy, 0.0, cy, 0.0, 0.0, 0.0, 0.0, 1.0];
    let rz = [cz, sz, 0.0, 0.0, -sz, cz, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    mmul(&mmul(&rx, &ry), &rz)
}

/// Looking from `eye` towards `at` (right-handed: the camera looks down -z).
fn look_at(eye: V3, at: V3, up: V3) -> M4 {
    let z = norm(sub(eye, at));
    let mut x = cross(up, z);
    if dot(x, x) < 1e-8 {
        x = cross([0.0, 0.0, 1.0], z);
    }
    let x = norm(x);
    let y = cross(z, x);
    [x[0], y[0], z[0], 0.0, x[1], y[1], z[1], 0.0, x[2], y[2], z[2], 0.0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1.0]
}

/// Perspective, with depth from 0 (near) to 1 (far), as Direct3D wants.
fn perspective(fov_deg: f32, aspect: f32, near: f32, far: f32) -> M4 {
    let ys = 1.0 / (fov_deg.to_radians() / 2.0).tan();
    let xs = ys / aspect;
    let q = far / (near - far);
    [xs, 0.0, 0.0, 0.0, 0.0, ys, 0.0, 0.0, 0.0, 0.0, q, -1.0, 0.0, 0.0, near * q, 0.0]
}

fn orthographic(w: f32, h: f32, near: f32, far: f32) -> M4 {
    let q = 1.0 / (near - far);
    [2.0 / w, 0.0, 0.0, 0.0, 0.0, 2.0 / h, 0.0, 0.0, 0.0, 0.0, q, 0.0, 0.0, 0.0, near * q, 1.0]
}

fn apply(p: V3, m: &M4) -> [f32; 4] {
    let v = [p[0], p[1], p[2], 1.0];
    let mut r = [0f32; 4];
    for (j, out) in r.iter_mut().enumerate() {
        *out = (0..4).map(|k| v[k] * m[k * 4 + j]).sum();
    }
    r
}

fn invert(m: &M4) -> M4 {
    // Gauss-Jordan elimination, in f64 for accuracy
    let mut a = [[0f64; 8]; 4];
    for i in 0..4 {
        for j in 0..4 {
            a[i][j] = m[i * 4 + j] as f64;
        }
        a[i][4 + i] = 1.0;
    }
    for c in 0..4 {
        let p = (c..4).max_by(|&x, &y| a[x][c].abs().partial_cmp(&a[y][c].abs()).unwrap()).unwrap();
        a.swap(c, p);
        let d = a[c][c];
        if d.abs() < 1e-30 {
            return IDENTITY;
        }
        for v in a[c].iter_mut() {
            *v /= d;
        }
        for r in 0..4 {
            if r != c {
                let f = a[r][c];
                let row = a[c];
                for (v, w) in a[r].iter_mut().zip(row.iter()) {
                    *v -= f * w;
                }
            }
        }
    }
    let mut r = [0f32; 16];
    for i in 0..4 {
        for j in 0..4 {
            r[i * 4 + j] = a[i][4 + j] as f32;
        }
    }
    r
}

// ---------- meshes ----------

/// Triangles to draw: each corner is a position, a normal (the way the
/// surface faces) and a place on a picture (u, v), 8 numbers in all.
pub struct Mesh {
    pub id: usize,
    pub verts: Vec<f32>,
    /// How many times its corners have been moved (by mesh(model, points)),
    /// so the copy on the graphics card is sent again.
    pub version: u64,
    /// For a model made from points: its triangles and places on a picture,
    /// kept so its corners can be moved.
    source: Option<Rc<Source>>,
    /// The middle and size of the box around it.
    centre: V3,
    radius: f32,
    size: V3,
}

struct Source {
    points: usize,
    tris: Vec<[usize; 3]>,
    uvs: Vec<[f32; 2]>,
    /// The faces that meet at each corner.
    meeting: Vec<Vec<usize>>,
}

fn meeting_of(points: usize, tris: &[[usize; 3]]) -> Vec<Vec<usize>> {
    let mut meeting: Vec<Vec<usize>> = vec![vec![]; points];
    for (ti, t) in tris.iter().enumerate() {
        for &k in t {
            meeting[k].push(ti);
        }
    }
    meeting
}

/// A model made from points, built again with its corners moved: like
/// `build`, but reusing what doesn't change.
fn rebuild(points: &[V3], src: &Source) -> Vec<f32> {
    let face: Vec<V3> = src.tris.iter().map(|t| cross(sub(points[t[1]], points[t[0]]), sub(points[t[2]], points[t[0]]))).collect();
    let unit: Vec<V3> = face.iter().map(|n| norm(*n)).collect();
    let mut out = Vec::with_capacity(src.tris.len() * 24);
    for (ti, t) in src.tris.iter().enumerate() {
        for &k in t {
            let mut sum = [0f32; 3];
            for &f in &src.meeting[k] {
                if dot(unit[f], unit[ti]) > 0.707 {
                    sum[0] += face[f][0];
                    sum[1] += face[f][1];
                    sum[2] += face[f][2];
                }
            }
            let uv = src.uvs.get(k).copied().unwrap_or([0.0, 0.0]);
            corner(&mut out, points[k], norm(sum), uv[0], 1.0 - uv[1]);
        }
    }
    out
}

/// Points as [x, y, z] rows, read straight from a table of numbers when
/// they are one (the quick way, for points made by tensor maths).
fn points_of(v: &Value, name: &str) -> Result<Vec<V3>, String> {
    if let Value::Tensor(t) = v {
        if t.shape.len() == 2 && t.shape[1] == 3 {
            return Ok(t.vals().chunks(3).map(|c| [c[0] as f32, c[1] as f32, c[2] as f32]).collect());
        }
    }
    Ok(rows(v, 3, name, "its points")?.iter().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect())
}

fn make_mesh(id: usize, verts: Vec<f32>) -> Mesh {
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for v in verts.chunks(8) {
        for k in 0..3 {
            lo[k] = lo[k].min(v[k]);
            hi[k] = hi[k].max(v[k]);
        }
    }
    if verts.is_empty() {
        (lo, hi) = ([0.0; 3], [0.0; 3]);
    }
    let centre = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0];
    let size = sub(hi, lo);
    let radius = dot(size, size).sqrt() / 2.0;
    Mesh { id, verts, version: 0, source: None, centre, radius, size }
}

fn corner(out: &mut Vec<f32>, p: V3, n: V3, u: f32, v: f32) {
    out.extend_from_slice(&[p[0], p[1], p[2], n[0], n[1], n[2], u, v]);
}

/// A cube 1 across, centred on the middle.
fn cube_mesh() -> Vec<f32> {
    let mut out = vec![];
    let faces: [(V3, V3, V3); 6] = [
        ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ];
    for (n, u, v) in faces {
        let at = |a: f32, b: f32| [n[0] * 0.5 + u[0] * a + v[0] * b, n[1] * 0.5 + u[1] * a + v[1] * b, n[2] * 0.5 + u[2] * a + v[2] * b];
        let (p00, p10, p11, p01) = (at(-0.5, -0.5), at(0.5, -0.5), at(0.5, 0.5), at(-0.5, 0.5));
        corner(&mut out, p00, n, 0.0, 1.0);
        corner(&mut out, p10, n, 1.0, 1.0);
        corner(&mut out, p11, n, 1.0, 0.0);
        corner(&mut out, p00, n, 0.0, 1.0);
        corner(&mut out, p11, n, 1.0, 0.0);
        corner(&mut out, p01, n, 0.0, 0.0);
    }
    out
}

/// A ball of radius 1.
fn sphere_mesh() -> Vec<f32> {
    let (slices, stacks) = (64, 32);
    let mut out = vec![];
    let at = |i: usize, j: usize| {
        let (u, v) = (i as f32 / slices as f32, j as f32 / stacks as f32);
        let (th, ph) = (u * std::f32::consts::TAU, v * std::f32::consts::PI);
        let p = [ph.sin() * th.cos(), ph.cos(), -ph.sin() * th.sin()];
        (p, u, v)
    };
    for j in 0..stacks {
        for i in 0..slices {
            let quad = [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)];
            for k in [0, 2, 1, 0, 3, 2] {
                let (p, u, v) = quad[k];
                corner(&mut out, p, p, u, v);
            }
        }
    }
    out
}

/// A cylinder (or, with `top` 0, a cone) of radius 1 and height 1, centred.
fn round_mesh(top: f32) -> Vec<f32> {
    let n = 64;
    let mut out = vec![];
    let slope = 1.0 - top; // how far the side leans in, over the height of 1
    let ring = |i: usize| {
        let t = i as f32 / n as f32 * std::f32::consts::TAU;
        (t.cos(), -t.sin())
    };
    for i in 0..n {
        let ((c0, s0), (c1, s1)) = (ring(i), ring(i + 1));
        let (u0, u1) = (i as f32 / n as f32, (i + 1) as f32 / n as f32);
        let side = |c: f32, s: f32| norm([c, slope, s]);
        let (b0, b1) = ([c0, -0.5, s0], [c1, -0.5, s1]);
        let (t0, t1) = ([c0 * top, 0.5, s0 * top], [c1 * top, 0.5, s1 * top]);
        let (n0, n1) = (side(c0, s0), side(c1, s1));
        corner(&mut out, b0, n0, u0, 1.0);
        corner(&mut out, b1, n1, u1, 1.0);
        corner(&mut out, t1, n1, u1, 0.0);
        corner(&mut out, b0, n0, u0, 1.0);
        corner(&mut out, t1, n1, u1, 0.0);
        corner(&mut out, t0, n0, u0, 0.0);
        // the bottom, and the top of a cylinder
        let down = [0.0, -1.0, 0.0];
        corner(&mut out, [0.0, -0.5, 0.0], down, 0.5, 0.5);
        corner(&mut out, b1, down, 0.5 + c1 / 2.0, 0.5 + s1 / 2.0);
        corner(&mut out, b0, down, 0.5 + c0 / 2.0, 0.5 + s0 / 2.0);
        if top > 0.0 {
            let up = [0.0, 1.0, 0.0];
            corner(&mut out, [0.0, 0.5, 0.0], up, 0.5, 0.5);
            corner(&mut out, t0, up, 0.5 + c0 / 2.0, 0.5 + s0 / 2.0);
            corner(&mut out, t1, up, 0.5 + c1 / 2.0, 0.5 + s1 / 2.0);
        }
    }
    out
}

/// A flat square 1 across, facing up.
fn plane_mesh() -> Vec<f32> {
    let mut out = vec![];
    let up = [0.0, 1.0, 0.0];
    let p = |x: f32, z: f32| [x, 0.0, z];
    corner(&mut out, p(-0.5, 0.5), up, 0.0, 1.0);
    corner(&mut out, p(0.5, 0.5), up, 1.0, 1.0);
    corner(&mut out, p(0.5, -0.5), up, 1.0, 0.0);
    corner(&mut out, p(-0.5, 0.5), up, 0.0, 1.0);
    corner(&mut out, p(0.5, -0.5), up, 1.0, 0.0);
    corner(&mut out, p(-0.5, -0.5), up, 0.0, 0.0);
    out
}

const CUBE: usize = 0;
const SPHERE: usize = 1;
const CYLINDER: usize = 2;
const CONE: usize = 3;
const PLANE: usize = 4;

thread_local! {
    /// Every mesh, by number: the five shapes, then models.
    static MESHES: RefCell<Vec<Rc<Mesh>>> = const { RefCell::new(vec![]) };
    /// Model files already read, by name.
    static LOADED: RefCell<Vec<(String, usize)>> = const { RefCell::new(vec![]) };
}

fn mesh(id: usize) -> Rc<Mesh> {
    MESHES.with(|m| {
        let mut m = m.borrow_mut();
        if m.is_empty() {
            for (i, v) in [cube_mesh(), sphere_mesh(), round_mesh(1.0), round_mesh(0.0), plane_mesh()].into_iter().enumerate() {
                m.push(Rc::new(make_mesh(i, v)));
            }
        }
        m[id].clone()
    })
}

fn add_mesh(verts: Vec<f32>, source: Option<Rc<Source>>) -> Result<Rc<Mesh>, String> {
    mesh(0);
    MESHES.with(|m| {
        let mut m = m.borrow_mut();
        if m.len() > 10_000 {
            return Err("there are already 10000 models: make each model once, before the loop, and draw it many times (mesh(model, points) moves a model's corners)".to_string());
        }
        let mut made = make_mesh(m.len(), verts);
        made.source = source;
        let made = Rc::new(made);
        m.push(made.clone());
        Ok(made)
    })
}

/// Triangles from corners and the triangles' corner numbers. Without
/// normals given, each corner's normal blends those of the faces meeting
/// there at gentle angles (under 45 degrees), so curved surfaces look smooth
/// and edges like a box's stay sharp.
fn build(points: &[V3], uvs: &[[f32; 2]], tris: &[[usize; 3]], normals: Option<&[V3]>, tri_normals: Option<&[[usize; 3]]>, tri_uvs: Option<&[[usize; 3]]>) -> Vec<f32> {
    let face: Vec<V3> = tris
        .iter()
        .map(|t| {
            let (a, b, c) = (points[t[0]], points[t[1]], points[t[2]]);
            cross(sub(b, a), sub(c, a)) // as long as the face is big
        })
        .collect();
    let unit: Vec<V3> = face.iter().map(|n| norm(*n)).collect();
    let mut meeting: Vec<Vec<usize>> = vec![vec![]; points.len()];
    for (ti, t) in tris.iter().enumerate() {
        for &k in t {
            meeting[k].push(ti);
        }
    }
    let given = |ti: usize, c: usize| match (normals, tri_normals) {
        (Some(ns), Some(tn)) => ns.get(tn[ti][c]).copied().map(norm),
        _ => None,
    };
    let mut out = Vec::with_capacity(tris.len() * 24);
    for (ti, t) in tris.iter().enumerate() {
        for c in 0..3 {
            let p = points[t[c]];
            let n = given(ti, c).unwrap_or_else(|| {
                let mut sum = [0f32; 3];
                for &f in &meeting[t[c]] {
                    if dot(unit[f], unit[ti]) > 0.707 {
                        for d in 0..3 {
                            sum[d] += face[f][d];
                        }
                    }
                }
                norm(sum)
            });
            let uv = match tri_uvs {
                Some(tu) => uvs.get(tu[ti][c]).copied().unwrap_or([0.0, 0.0]),
                None => uvs.get(t[c]).copied().unwrap_or([0.0, 0.0]),
            };
            corner(&mut out, p, n, uv[0], 1.0 - uv[1]);
        }
    }
    out
}

/// A Wavefront .obj file: corners (v), places on a picture (vt), normals
/// (vn) and faces (f), which may have more than three corners.
fn read_obj(src: &str) -> Result<Vec<f32>, String> {
    let (mut vs, mut vts, mut vns) = (vec![], vec![], vec![]);
    let (mut tris, mut tri_uv, mut tri_n) = (vec![], vec![], vec![]);
    let (mut has_uv, mut has_n) = (true, true);
    for (ln, line) in src.lines().enumerate() {
        let mut w = line.split_whitespace();
        let bad = || format!("line {} isn't understood: {}", ln + 1, line.trim());
        let nums = |w: std::str::SplitWhitespace, n: usize| -> Result<Vec<f32>, String> {
            let v: Vec<f32> = w.take(n).map(|x| x.parse::<f32>()).collect::<Result<_, _>>().map_err(|_| bad())?;
            Ok(v)
        };
        match w.next() {
            Some("v") => {
                let p = nums(w, 3)?;
                if p.len() < 3 {
                    return Err(bad());
                }
                vs.push([p[0], p[1], p[2]]);
            }
            Some("vt") => {
                let p = nums(w, 2)?;
                vts.push([p.first().copied().unwrap_or(0.0), p.get(1).copied().unwrap_or(0.0)]);
            }
            Some("vn") => {
                let p = nums(w, 3)?;
                if p.len() < 3 {
                    return Err(bad());
                }
                vns.push([p[0], p[1], p[2]]);
            }
            Some("f") => {
                let mut face: Vec<(usize, Option<usize>, Option<usize>)> = vec![];
                for c in w {
                    let parts: Vec<&str> = c.split('/').collect();
                    // 1 is the first; -1 the last so far
                    let index = |s: &str, count: usize| -> Result<Option<usize>, String> {
                        if s.is_empty() {
                            return Ok(None);
                        }
                        let i: i64 = s.parse().map_err(|_| bad())?;
                        let k = if i < 0 { count as i64 + i } else { i - 1 };
                        if k < 0 || k >= count as i64 {
                            return Err(format!("line {} refers to a corner that isn't there", ln + 1));
                        }
                        Ok(Some(k as usize))
                    };
                    let v = index(parts[0], vs.len())?.ok_or_else(bad)?;
                    let t = match parts.get(1) {
                        Some(s) => index(s, vts.len())?,
                        None => None,
                    };
                    let n = match parts.get(2) {
                        Some(s) => index(s, vns.len())?,
                        None => None,
                    };
                    face.push((v, t, n));
                }
                if face.len() < 3 {
                    return Err(bad());
                }
                for k in 1..face.len() - 1 {
                    let c = [face[0], face[k], face[k + 1]];
                    tris.push([c[0].0, c[1].0, c[2].0]);
                    has_uv &= c.iter().all(|x| x.1.is_some());
                    has_n &= c.iter().all(|x| x.2.is_some());
                    tri_uv.push([c[0].1.unwrap_or(0), c[1].1.unwrap_or(0), c[2].1.unwrap_or(0)]);
                    tri_n.push([c[0].2.unwrap_or(0), c[1].2.unwrap_or(0), c[2].2.unwrap_or(0)]);
                }
            }
            _ => {}
        }
    }
    if tris.is_empty() {
        return Err("it has no faces".into());
    }
    let (normals, tri_normals) = if has_n && !vns.is_empty() { (Some(&vns[..]), Some(&tri_n[..])) } else { (None, None) };
    let tri_uvs = if has_uv && !vts.is_empty() { Some(&tri_uv[..]) } else { None };
    Ok(build(&vs, &vts, &tris, normals, tri_normals, tri_uvs))
}

// ---------- the scene ----------

/// How a shape looks, and how it is placed.
struct Look {
    /// Red, green, blue (in light's own units, not as displayed) and alpha.
    colour: [f32; 4],
    shine: f32,
    metal: f32,
    glow: f32,
    ripples: f32,
    tiles: f32,
    picture: Option<usize>,
    turn: V3,
    scale: V3,
    casts: bool,
}

#[cfg_attr(not(windows), allow(dead_code))]
/// One shape or model to draw.
pub struct Item {
    mesh: Rc<Mesh>,
    world: M4,
    normal: M4,
    colour: [f32; 4],
    mat: [f32; 4],
    ripples: f32,
    picture: Option<usize>,
    casts: bool,
    centre: V3,
    radius: f32,
}

struct Lamp {
    pos: V3,
    colour: V3,
    reach: f32,
}

struct Scene {
    eye: V3,
    at: V3,
    fov: f32,
    near: f32,
    far: f32,
    /// Towards the sun, and its light.
    sun: Option<(V3, V3)>,
    shadows: bool,
    /// Top, horizon and ground colours of the sky (in light units), and
    /// whether it is drawn behind everything.
    sky: [V3; 3],
    sky_shown: bool,
    fog: Option<(V3, f32, f32)>,
    items: Vec<Item>,
    lamps: Vec<Lamp>,
    /// Shapes waiting to be drawn.
    pending: bool,
}

fn linear(c: f64) -> f32 {
    ((c / 255.0).clamp(0.0, 1.0) as f32).powf(2.2)
}

thread_local! {
    static SCENE: RefCell<Scene> = RefCell::new(Scene {
        eye: [0.0, 3.0, 8.0],
        at: [0.0, 0.5, 0.0],
        fov: 60.0,
        near: 0.1,
        far: 1000.0,
        sun: Some((norm([0.5, 1.0, 0.35]), [2.4, 2.3, 2.16])),
        shadows: true,
        sky: [[linear(140.0), linear(185.0), linear(235.0)], [linear(222.0), linear(232.0), linear(242.0)], [linear(120.0), linear(110.0), linear(96.0)]],
        sky_shown: false,
        fog: None,
        items: vec![],
        lamps: vec![],
        pending: false,
    });
}

/// A colour (any form 2D drawing takes) in light units, with its alpha.
fn colour3(v: &Value, name: &str) -> Result<[f32; 4], String> {
    let (r, g, b, a) = crate::gfx::colour(v, name)?.straight();
    Ok([linear(r), linear(g), linear(b), a as f32])
}

fn coord(v: &Value, name: &str) -> Result<f32, String> {
    let x = num(v, name)?;
    if !x.is_finite() {
        return Err(format!("{} needs an ordinary number, not {}", name, x));
    }
    Ok(x as f32)
}

fn three(v: &Value, name: &str, what: &str) -> Result<V3, String> {
    let parts: Option<Vec<f64>> = match v {
        Value::List(l) => l.iter().map(|x| x.scalar()).collect(),
        Value::Tensor(t) => Some(t.vals().clone()),
        _ => None,
    };
    match parts {
        Some(p) if p.len() == 3 && p.iter().all(|x| x.is_finite()) => Ok([p[0] as f32, p[1] as f32, p[2] as f32]),
        _ => Err(format!("{} needs {} as [x, y, z], not {}", name, what, v.show())),
    }
}

const LOOK_OPTIONS: &[&str] = &["colour", "picture", "tiles", "shine", "metal", "glow", "ripples", "turn", "scale", "shadow"];

/// A shape's look: a colour, or a map of options.
fn look(v: Option<&Value>, name: &str) -> Result<Look, String> {
    let mut l = Look { colour: [linear(200.0), linear(200.0), linear(200.0), 1.0], shine: 0.3, metal: 0.0, glow: 0.0, ripples: 0.0, tiles: 1.0, picture: None, turn: [0.0; 3], scale: [1.0; 3], casts: true };
    let Some(v) = v else { return Ok(l) };
    let Value::Map(m) = v else {
        l.colour = colour3(v, name)?;
        return Ok(l);
    };
    let unit = |x: &Value| -> Result<f32, String> { Ok(num(x, name)?.clamp(0.0, 1.0) as f32) };
    for (k, x) in m.iter() {
        match &**k {
            "colour" | "color" => l.colour = colour3(x, name)?,
            "picture" => {
                let id = crate::gfx::picture_id(x, name)?;
                l.picture = Some(id);
                if !m.iter().any(|(k, _)| &**k == "colour" || &**k == "color") {
                    l.colour = [1.0, 1.0, 1.0, 1.0];
                }
            }
            "shine" => l.shine = unit(x)?,
            "metal" => l.metal = unit(x)?,
            "glow" => l.glow = num(x, name)?.clamp(0.0, 100.0) as f32,
            "ripples" => l.ripples = num(x, name)?.clamp(0.0, 10.0) as f32,
            "tiles" => l.tiles = coord(x, name)?,
            "turn" => {
                l.turn = match x {
                    Value::Num(_) => [0.0, coord(x, name)?, 0.0],
                    _ => three(x, name, "turn")?,
                }
            }
            "scale" => {
                l.scale = match x {
                    Value::Num(_) => {
                        let s = coord(x, name)?;
                        [s, s, s]
                    }
                    _ => three(x, name, "scale")?,
                }
            }
            "shadow" => {
                l.casts = match x {
                    Value::Bool(b) => *b,
                    _ => return Err(format!("{}'s shadow option is yes or no, not {}", name, x.show())),
                }
            }
            other => return Err(format!("{} doesn't know the option \"{}\"; it knows {}", name, other, LOOK_OPTIONS.join(", "))),
        }
    }
    Ok(l)
}

/// Adds a mesh to the scene, `size` across in each direction, centred on
/// `pos`.
fn place(mesh: Rc<Mesh>, pos: V3, size: V3, l: Look) {
    let s = [size[0] * l.scale[0], size[1] * l.scale[1], size[2] * l.scale[2]];
    let r = turning(l.turn);
    let world = mmul(&mmul(&scaling(s), &r), &translate(pos));
    // normals turn with the shape, but a stretch squeezes them the other way
    let inv = |x: f32| if x.abs() > 1e-12 { 1.0 / x } else { 0.0 };
    let normal = mmul(&scaling([inv(s[0]), inv(s[1]), inv(s[2])]), &r);
    let c = apply(mesh.centre, &world);
    let radius = mesh.radius * s.iter().fold(0f32, |a, b| a.max(b.abs()));
    let item = Item {
        world,
        normal,
        colour: l.colour,
        mat: [l.shine, l.metal, l.glow, l.tiles],
        ripples: l.ripples,
        picture: l.picture,
        casts: l.casts,
        centre: [c[0], c[1], c[2]],
        radius,
        mesh,
    };
    SCENE.with(|sc| {
        let mut sc = sc.borrow_mut();
        sc.items.push(item);
        sc.pending = true;
    });
}

fn inputs(a: &[Value], lo: usize, hi: usize, name: &str, what: &str) -> Result<(), String> {
    if a.len() < lo || a.len() > hi {
        return Err(format!("{} takes {}, but was given {} input{}", name, what, a.len(), if a.len() == 1 { "" } else { "s" }));
    }
    Ok(())
}

fn position(a: &[Value], name: &str) -> Result<V3, String> {
    Ok([coord(&a[0], name)?, coord(&a[1], name)?, coord(&a[2], name)?])
}

fn size(v: &Value, name: &str) -> Result<f32, String> {
    let s = coord(v, name)?;
    if s < 0.0 {
        return Err(format!("{} needs a size of 0 or more, not {}", name, v.show()));
    }
    Ok(s)
}

// ---------- the built-in functions ----------

/// camera(x, y, z, look_x, look_y, look_z [, {fov, near, far}]): where the
/// camera is and what it looks at.
pub fn b_camera(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 6, 7, "camera", "where it is (x, y, z) and the point it looks at, and optionally a map of options")?;
    let eye = position(&a[..3], "camera")?;
    let at = position(&a[3..6], "camera")?;
    let (mut fov, mut near, mut far) = SCENE.with(|s| {
        let s = s.borrow();
        (s.fov, s.near, s.far)
    });
    if let Some(o) = a.get(6) {
        let Value::Map(m) = o else {
            return Err(format!("camera's seventh input is a map of options, like {{fov: 70}}, not {}", o.show()));
        };
        for (k, v) in m.iter() {
            match &**k {
                "fov" => fov = coord(v, "camera")?.clamp(1.0, 170.0),
                "near" => near = coord(v, "camera")?.max(1e-4),
                "far" => far = coord(v, "camera")?,
                other => return Err(format!("camera doesn't know the option \"{}\"; it knows fov, near and far", other)),
            }
        }
    }
    if far <= near {
        return Err("camera's far distance must be more than its near one".into());
    }
    SCENE.with(|s| {
        let mut s = s.borrow_mut();
        (s.eye, s.at, s.fov, s.near, s.far) = (eye, at, fov, near, far);
    });
    Ok(Value::Nothing)
}

/// sun(x, y, z [, colour [, strength]]): sunlight coming from that
/// direction; sun() turns it off.
pub fn b_sun(_: &mut Interp, a: Vec<Value>) -> Res {
    if a.is_empty() {
        SCENE.with(|s| s.borrow_mut().sun = None);
        return Ok(Value::Nothing);
    }
    inputs(&a, 3, 5, "sun", "the direction the sunlight comes from (x, y, z), and optionally a colour and a strength")?;
    let dir = position(&a, "sun")?;
    if dot(dir, dir) == 0.0 {
        return Err("sun needs a direction that isn't [0, 0, 0]".into());
    }
    let c = match a.get(3) {
        Some(v) => colour3(v, "sun")?,
        None => [1.0, 0.96, 0.9, 1.0],
    };
    let k = match a.get(4) {
        Some(v) => coord(v, "sun")?.max(0.0),
        None => 1.0,
    } * 2.4;
    SCENE.with(|s| s.borrow_mut().sun = Some((norm(dir), [c[0] * k, c[1] * k, c[2] * k])));
    Ok(Value::Nothing)
}

/// lamp(x, y, z [, colour [, strength [, reach]]]): a light at a point, for
/// this frame (up to 8).
pub fn b_lamp(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 3, 6, "lamp", "where it is (x, y, z), and optionally a colour, a strength and how far it reaches")?;
    let pos = position(&a, "lamp")?;
    let c = match a.get(3) {
        Some(v) => colour3(v, "lamp")?,
        None => [1.0, 0.85, 0.65, 1.0],
    };
    let k = match a.get(4) {
        Some(v) => coord(v, "lamp")?.max(0.0),
        None => 1.0,
    } * 12.0;
    let reach = match a.get(5) {
        Some(v) => coord(v, "lamp")?.max(0.01),
        None => 12.0,
    };
    SCENE.with(|s| {
        let mut s = s.borrow_mut();
        if s.lamps.len() >= 8 {
            return Err("there can be up to 8 lamps at a time".to_string());
        }
        s.lamps.push(Lamp { pos, colour: [c[0] * k, c[1] * k, c[2] * k], reach });
        s.pending = true;
        Ok(())
    })?;
    Ok(Value::Nothing)
}

/// sky([top [, horizon [, ground]]]): the sky drawn behind everything, which
/// also lights the scene softly; sky() leaves the background as it is.
pub fn b_sky(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 0, 3, "sky", "a colour for the top, and optionally for the horizon and the ground")?;
    if a.is_empty() {
        SCENE.with(|s| s.borrow_mut().sky_shown = false);
        return Ok(Value::Nothing);
    }
    let top = colour3(&a[0], "sky")?;
    let hor = match a.get(1) {
        Some(v) => colour3(v, "sky")?,
        None => {
            let w = |c: f32| c + (1.0 - c) * 0.55;
            [w(top[0]), w(top[1]), w(top[2]), 1.0]
        }
    };
    let gnd = match a.get(2) {
        Some(v) => colour3(v, "sky")?,
        None => [hor[0] * 0.35, hor[1] * 0.32, hor[2] * 0.28, 1.0],
    };
    let rgb = |c: [f32; 4]| [c[0], c[1], c[2]];
    SCENE.with(|s| {
        let mut s = s.borrow_mut();
        s.sky = [rgb(top), rgb(hor), rgb(gnd)];
        s.sky_shown = true;
    });
    Ok(Value::Nothing)
}

/// fog([colour,] near, far): things fade into the fog from `near` away to
/// `far` away; fog() clears it.
pub fn b_fog(_: &mut Interp, a: Vec<Value>) -> Res {
    if a.is_empty() {
        SCENE.with(|s| s.borrow_mut().fog = None);
        return Ok(Value::Nothing);
    }
    inputs(&a, 2, 3, "fog", "optionally a colour, then how far away the fog starts and where it is thickest")?;
    let (c, rest) = if a.len() == 3 { (Some(colour3(&a[0], "fog")?), &a[1..]) } else { (None, &a[..]) };
    let (near, far) = (coord(&rest[0], "fog")?, coord(&rest[1], "fog")?);
    if far <= near {
        return Err("fog's far distance must be more than its near one".into());
    }
    SCENE.with(|s| {
        let mut s = s.borrow_mut();
        let col = match c {
            Some(c) => [c[0], c[1], c[2]],
            None => s.sky[1],
        };
        s.fog = Some((col, near, far));
    });
    Ok(Value::Nothing)
}

/// shadows(yes or no): whether the sun casts shadows (it does unless told
/// not to).
pub fn b_shadows(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 1, "shadows")?;
    let Value::Bool(on) = a[0] else {
        return Err(format!("shadows needs yes or no, not {}", a[0].show()));
    };
    SCENE.with(|s| s.borrow_mut().shadows = on);
    Ok(Value::Nothing)
}

/// cube(x, y, z, size [, look])
pub fn b_cube(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 4, 5, "cube", "its middle (x, y, z) and its size, and optionally a colour or a map of options")?;
    let p = position(&a, "cube")?;
    let s = size(&a[3], "cube")?;
    place(mesh(CUBE), p, [s, s, s], look(a.get(4), "cube")?);
    Ok(Value::Nothing)
}

/// box(x, y, z, width, height, depth [, look])
pub fn b_box(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 6, 7, "box", "its middle (x, y, z), its width, height and depth, and optionally a colour or a map of options")?;
    let p = position(&a, "box")?;
    let s = [size(&a[3], "box")?, size(&a[4], "box")?, size(&a[5], "box")?];
    place(mesh(CUBE), p, s, look(a.get(6), "box")?);
    Ok(Value::Nothing)
}

/// sphere(x, y, z, radius [, look])
pub fn b_sphere(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 4, 5, "sphere", "its middle (x, y, z) and its radius, and optionally a colour or a map of options")?;
    let p = position(&a, "sphere")?;
    let r = size(&a[3], "sphere")?;
    place(mesh(SPHERE), p, [r, r, r], look(a.get(4), "sphere")?);
    Ok(Value::Nothing)
}

/// cylinder(x, y, z, radius, height [, look])
pub fn b_cylinder(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 5, 6, "cylinder", "its middle (x, y, z), its radius and its height, and optionally a colour or a map of options")?;
    let p = position(&a, "cylinder")?;
    let (r, h) = (size(&a[3], "cylinder")?, size(&a[4], "cylinder")?);
    place(mesh(CYLINDER), p, [r, h, r], look(a.get(5), "cylinder")?);
    Ok(Value::Nothing)
}

/// cone(x, y, z, radius, height [, look])
pub fn b_cone(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 5, 6, "cone", "its middle (x, y, z), its radius and its height, and optionally a colour or a map of options")?;
    let p = position(&a, "cone")?;
    let (r, h) = (size(&a[3], "cone")?, size(&a[4], "cone")?);
    place(mesh(CONE), p, [r, h, r], look(a.get(5), "cone")?);
    Ok(Value::Nothing)
}

/// plane(x, y, z, width, depth [, look]): a flat rectangle facing up, such
/// as the ground.
pub fn b_plane(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 5, 6, "plane", "its middle (x, y, z), its width and depth, and optionally a colour or a map of options")?;
    let p = position(&a, "plane")?;
    let (w, d) = (size(&a[3], "plane")?, size(&a[4], "plane")?);
    place(mesh(PLANE), p, [w, 1.0, d], look(a.get(5), "plane")?);
    Ok(Value::Nothing)
}

fn model_value(m: &Mesh) -> Value {
    Value::Map(Rc::new(vec![
        ("mesh".into(), Value::Num(m.id as f64)),
        ("size".into(), crate::value::pack_list(m.size.iter().map(|x| Value::Num(*x as f64)).collect())),
        ("triangles".into(), Value::Num((m.verts.len() / 24) as f64)),
    ]))
}

fn rows(v: &Value, n: usize, name: &str, what: &str) -> Result<Vec<Vec<f64>>, String> {
    let bad = || format!("{} needs {} as a list of [{}] rows", name, what, ["x", "y", "z"][..n].join(", "));
    match v {
        Value::Tensor(t) if t.shape.len() == 2 && t.shape[1] == n => Ok(t.vals().chunks(n).map(|c| c.to_vec()).collect()),
        Value::List(l) => l
            .iter()
            .map(|r| match r {
                Value::Tensor(t) if t.vals().len() == n => Ok(t.vals().clone()),
                Value::List(x) if x.len() == n => x.iter().map(|y| y.scalar().ok_or_else(bad)).collect(),
                _ => Err(bad()),
            })
            .collect(),
        _ => Err(bad()),
    }
}

/// mesh(file): reads a model from a Wavefront .obj file; mesh(points,
/// triangles [, places]): a model made of corners [x, y, z] and triangles
/// [a, b, c] (corner numbers, from 0); mesh(model, points): the same model
/// with its corners moved (water, flags, cloth), keeping its triangles.
pub fn b_mesh(_: &mut Interp, a: Vec<Value>) -> Res {
    inputs(&a, 1, 3, "mesh", "an .obj file's name, or a list of points and a list of triangles")?;
    if a.len() == 2 && is_model(&a[0]) {
        return move_corners(&a[0], &a[1]);
    }
    if a.len() == 1 {
        let file = text(&a[0], "mesh")?.to_string();
        if let Some(id) = LOADED.with(|l| l.borrow().iter().find(|(f, _)| *f == file).map(|(_, i)| *i)) {
            return Ok(model_value(&mesh(id)));
        }
        let src = match crate::builtins::embedded_file(&file) {
            Some(Some(b)) => String::from_utf8_lossy(b).into_owned(),
            Some(None) => return Err(format!("mesh couldn't find \"{}\": no file of that name was embedded", file)),
            None => std::fs::read_to_string(&file).map_err(|e| format!("couldn't read the model \"{}\": {}", file, e))?,
        };
        let verts = read_obj(&src).map_err(|e| format!("couldn't read the model \"{}\": {}", file, e))?;
        let m = add_mesh(verts, None)?;
        LOADED.with(|l| l.borrow_mut().push((file, m.id)));
        return Ok(model_value(&m));
    }
    let pts = points_of(&a[0], "mesh")?;
    let mut tris = vec![];
    for t in rows(&a[1], 3, "mesh", "its triangles")? {
        let mut c = [0usize; 3];
        for k in 0..3 {
            if t[k] < 0.0 || t[k] != t[k].trunc() || t[k] as usize >= pts.len() {
                return Err(format!("mesh's triangles name corners by number, from 0 to {}, not {}", pts.len().saturating_sub(1), crate::value::fmt_num(t[k])));
            }
            c[k] = t[k] as usize;
        }
        tris.push(c);
    }
    let uvs: Vec<[f32; 2]> = match a.get(2) {
        Some(v) => {
            let u = rows(v, 2, "mesh", "its places on a picture")?;
            if u.len() != pts.len() {
                return Err(format!("mesh needs one place on the picture for each of its {} points", pts.len()));
            }
            u.iter().map(|p| [p[0] as f32, 1.0 - p[1] as f32]).collect()
        }
        None => vec![],
    };
    let verts = build(&pts, &uvs, &tris, None, None, None);
    let meeting = meeting_of(pts.len(), &tris);
    let m = add_mesh(verts, Some(Rc::new(Source { points: pts.len(), tris, uvs, meeting })))?;
    Ok(model_value(&m))
}

/// The number of a model made by `mesh`.
fn model_id(v: &Value, name: &str) -> Result<usize, String> {
    let id = match v {
        Value::Map(m) => m.iter().find(|(k, _)| &**k == "mesh").and_then(|(_, v)| v.scalar()),
        _ => None,
    };
    let count = MESHES.with(|m| m.borrow().len());
    match id {
        Some(n) if n >= 0.0 && n == n.trunc() && (n as usize) < count => Ok(n as usize),
        _ => Err(format!("{} needs a model made by mesh(...), not {}", name, v.show())),
    }
}

/// mesh(model, points): moves the corners of a model made from points.
fn move_corners(model: &Value, points: &Value) -> Res {
    let id = model_id(model, "mesh")?;
    let old = mesh(id);
    let Some(src) = old.source.clone() else {
        return Err("mesh can move the corners only of a model made from points and triangles, not of a shape or a model read from a file".into());
    };
    let pts = points_of(points, "mesh")?;
    if pts.len() != src.points {
        return Err(format!("mesh needs the model's {} points, moved, not {}", src.points, pts.len()));
    }
    let mut m = make_mesh(id, rebuild(&pts, &src));
    m.version = old.version + 1;
    m.source = Some(src);
    let m = Rc::new(m);
    MESHES.with(|all| all.borrow_mut()[id] = m.clone());
    Ok(model_value(&m))
}

/// draw(model, x, y, z [, look]): a model made by `mesh`, its middle at
/// (x, y, z) (its own origin, for a model made of points).
pub fn b_draw_model(a: Vec<Value>) -> Res {
    inputs(&a, 4, 5, "draw", "a model, where to put it (x, y, z), and optionally a colour or a map of options")?;
    let id = model_id(&a[0], "draw")?;
    let p = position(&a[1..4], "draw")?;
    place(mesh(id), p, [1.0, 1.0, 1.0], look(a.get(4), "draw")?);
    Ok(Value::Nothing)
}

/// Whether a value is a model made by `mesh`.
pub fn is_model(v: &Value) -> bool {
    matches!(v, Value::Map(m) if m.iter().any(|(k, _)| &**k == "mesh"))
}

/// The camera's view and projection for a picture of this size.
fn view_proj(s: &Scene, w: usize, h: usize) -> M4 {
    let view = look_at(s.eye, s.at, [0.0, 1.0, 0.0]);
    let proj = perspective(s.fov, w as f32 / h.max(1) as f32, s.near, s.far);
    mmul(&view, &proj)
}

/// to_screen(x, y, z): where a point in the 3D world shows in the picture,
/// as [x, y], or nothing if it is behind the camera.
pub fn b_to_screen(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 3, "to_screen")?;
    let p = position(&a, "to_screen")?;
    let (w, h) = crate::gfx::target_size("to_screen")?;
    let c = SCENE.with(|s| apply(p, &view_proj(&s.borrow(), w, h)));
    if c[3] <= 1e-6 {
        return Ok(Value::Nothing);
    }
    let (x, y) = (c[0] / c[3], c[1] / c[3]);
    Ok(crate::value::pack_list(vec![Value::Num(((x * 0.5 + 0.5) * w as f32) as f64), Value::Num(((0.5 - y * 0.5) * h as f32) as f64)]))
}

/// render(): draws the 3D scene described since the last render into the
/// picture (the window's, or one chosen with canvas), over what is there.
pub fn b_render(_: &mut Interp, a: Vec<Value>) -> Res {
    want(&a, 0, "render")?;
    render()?;
    Ok(Value::Nothing)
}

/// Renders the scene if shapes are waiting (`frame` and `screenshot` call
/// this, so a 3D program needn't call render itself).
pub fn flush() -> Result<(), String> {
    if SCENE.with(|s| s.borrow().pending) {
        render()?;
    }
    Ok(())
}

/// The constants every shader sees, laid out as the shaders expect.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FrameConsts {
    view_proj: M4,
    light_vp: M4,
    inv_view_proj: M4,
    cam: [f32; 4],
    sun_dir: [f32; 4],
    sun_col: [f32; 4],
    sky_top: [f32; 4],
    sky_hor: [f32; 4],
    sky_gnd: [f32; 4],
    fog_col: [f32; 4],
    params: [f32; 4],
    shadow_info: [f32; 4],
    lamp_pos: [[f32; 4]; 8],
    lamp_col: [[f32; 4]; 8],
}

/// The constants for one shape.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ItemConsts {
    world: M4,
    normal: M4,
    colour: [f32; 4],
    mat: [f32; 4],
    flags: [f32; 4],
}

pub const SHADOW_SIZE: u32 = 2048;

/// Seconds since the program's first 3D picture, for things that move by
/// themselves on the graphics card (ripples).
fn seconds() -> f32 {
    thread_local! {
        static START: std::time::Instant = std::time::Instant::now();
    }
    START.with(|s| s.elapsed().as_secs_f32())
}

fn render() -> Result<(), String> {
    let scene = SCENE.with(|s| {
        let mut s = s.borrow_mut();
        let items = std::mem::take(&mut s.items);
        let lamps = std::mem::take(&mut s.lamps);
        s.pending = false;
        (items, lamps)
    });
    let (items, lamps) = scene;
    let (w, h) = crate::gfx::target_size("render")?;
    let (frame, sky_shown, shadows) = SCENE.with(|s| {
        let s = s.borrow();
        let vp = view_proj(&s, w, h);
        let mut f = FrameConsts {
            view_proj: vp,
            light_vp: IDENTITY,
            inv_view_proj: invert(&vp),
            cam: [s.eye[0], s.eye[1], s.eye[2], 1.0],
            sun_dir: [0.0; 4],
            sun_col: [0.0; 4],
            sky_top: [s.sky[0][0], s.sky[0][1], s.sky[0][2], 1.0],
            sky_hor: [s.sky[1][0], s.sky[1][1], s.sky[1][2], 1.0],
            sky_gnd: [s.sky[2][0], s.sky[2][1], s.sky[2][2], 1.0],
            fog_col: [0.0; 4],
            params: [0.0, 1.0, seconds(), lamps.len() as f32],
            // the last but one: how wide a pixel is, for each unit away
            shadow_info: [0.0, 1.0 / SHADOW_SIZE as f32, 2.0 * (s.fov.to_radians() / 2.0).tan() / h.max(1) as f32, 0.0],
            lamp_pos: [[0.0; 4]; 8],
            lamp_col: [[0.0; 4]; 8],
        };
        let mut shadows = false;
        if let Some((d, c)) = s.sun {
            f.sun_dir = [d[0], d[1], d[2], 1.0];
            f.sun_col = [c[0], c[1], c[2], 0.0];
            let casters: Vec<&Item> = items.iter().filter(|i| i.casts && i.colour[3] >= 1.0).collect();
            if s.shadows && !casters.is_empty() {
                // The sun's view, fitted around what it lights: the whole
                // scene, or for a big one, the part in front of the camera.
                let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
                for i in &items {
                    for k in 0..3 {
                        lo[k] = lo[k].min(i.centre[k] - i.radius);
                        hi[k] = hi[k].max(i.centre[k] + i.radius);
                    }
                }
                let mut centre = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0];
                let mut r = dot(sub(hi, lo), sub(hi, lo)).sqrt() / 2.0;
                if r > 60.0 {
                    let ahead = norm(sub(s.at, s.eye));
                    centre = [s.eye[0] + ahead[0] * 25.0, s.eye[1] + ahead[1] * 25.0, s.eye[2] + ahead[2] * 25.0];
                    r = 40.0;
                }
                let r = r.max(0.5);
                let eye = [centre[0] + d[0] * r * 2.0, centre[1] + d[1] * r * 2.0, centre[2] + d[2] * r * 2.0];
                let view = look_at(eye, centre, [0.0, 1.0, 0.0]);
                f.light_vp = mmul(&view, &orthographic(2.0 * r, 2.0 * r, 0.0, 4.0 * r));
                f.sun_col[3] = 1.0;
                f.shadow_info[0] = 2.0 * r / SHADOW_SIZE as f32;
                shadows = true;
            }
        }
        if let Some((c, n, fr)) = s.fog {
            f.fog_col = [c[0], c[1], c[2], 1.0];
            f.params[0] = n;
            f.params[1] = fr;
        }
        for (k, l) in lamps.iter().enumerate() {
            f.lamp_pos[k] = [l.pos[0], l.pos[1], l.pos[2], l.reach];
            f.lamp_col[k] = [l.colour[0], l.colour[1], l.colour[2], 0.0];
        }
        (f, s.sky_shown, shadows)
    });
    if items.is_empty() && !sky_shown {
        return Ok(());
    }
    let eye = SCENE.with(|s| s.borrow().eye);
    draw_scene(w, h, &frame, &items, sky_shown, shadows, eye)
}

#[cfg(not(windows))]
fn draw_scene(_: usize, _: usize, _: &FrameConsts, _: &[Item], _: bool, _: bool, _: V3) -> Result<(), String> {
    Err("3D graphics work on Windows for now (they use Direct3D); macOS and Linux are coming".into())
}

#[cfg(windows)]
fn draw_scene(w: usize, h: usize, frame: &FrameConsts, items: &[Item], sky: bool, shadows: bool, eye: V3) -> Result<(), String> {
    // see-through shapes go last, furthest first
    let mut order: Vec<usize> = (0..items.len()).collect();
    let dist = |i: &Item| dot(sub(i.centre, eye), sub(i.centre, eye));
    order.sort_by(|&a, &b| {
        let (ia, ib) = (&items[a], &items[b]);
        let (ta, tb) = (ia.colour[3] < 1.0, ib.colour[3] < 1.0);
        ta.cmp(&tb).then_with(|| if ta { dist(ib).partial_cmp(&dist(ia)).unwrap_or(std::cmp::Ordering::Equal) } else { std::cmp::Ordering::Equal })
    });
    let draws: Vec<d3d::Draw> = order
        .iter()
        .map(|&k| {
            let i = &items[k];
            let see_through = i.colour[3] < 1.0;
            d3d::Draw {
                mesh: i.mesh.as_ref(),
                consts: ItemConsts {
                    world: i.world,
                    normal: i.normal,
                    colour: i.colour,
                    mat: i.mat,
                    flags: [i.picture.is_some() as u32 as f32, see_through as u32 as f32, i.ripples, 0.0],
                },
                picture: i.picture,
                casts: i.casts && !see_through,
                see_through,
            }
        })
        .collect();
    let px = match d3d::with_renderer(|r| r.render(w, h, frame, &draws, sky, shadows)) {
        Ok(px) => px,
        Err(e) => {
            d3d::reset();
            return Err(e);
        }
    };
    crate::gfx::paint_rendered(w, h, px, sky)
}

/// The name of what draws 3D: a graphics card, or Windows' software renderer.
#[cfg(windows)]
pub fn renderer_name() -> Option<String> {
    d3d::with_renderer(|r| Ok(r.name.clone())).ok()
}

#[cfg(not(windows))]
pub fn renderer_name() -> Option<String> {
    None
}

// ---------- Direct3D 11 ----------

#[cfg(windows)]
mod d3d {
    use super::{FrameConsts, ItemConsts, Mesh, SHADOW_SIZE};
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::ffi::c_void;

    type Ptr = *mut c_void;
    type HResult = i32;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryW(name: *const u16) -> Ptr;
        fn GetProcAddress(module: Ptr, name: *const u8) -> Ptr;
    }

    /// Calls method `index` of a COM object's table of functions.
    unsafe fn method<F: Copy>(obj: Ptr, index: usize) -> F {
        let table = unsafe { *(obj as *const *const Ptr) };
        let f = unsafe { *table.add(index) };
        unsafe { std::mem::transmute_copy(&f) }
    }

    /// A COM object, released when dropped.
    pub struct Com(Ptr);

    impl Drop for Com {
        fn drop(&mut self) {
            if !self.0.is_null() {
                let f: unsafe extern "system" fn(Ptr) -> u32 = unsafe { method(self.0, 2) };
                unsafe { f(self.0) };
            }
        }
    }

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain([0]).collect()
    }

    fn load(dll: &str, func: &str) -> Option<Ptr> {
        let m = unsafe { LoadLibraryW(wide(dll).as_ptr()) };
        if m.is_null() {
            return None;
        }
        let name: Vec<u8> = func.bytes().chain([0]).collect();
        let f = unsafe { GetProcAddress(m, name.as_ptr()) };
        (!f.is_null()).then_some(f)
    }

    #[repr(C)]
    struct BufferDesc {
        byte_width: u32,
        usage: u32,
        bind: u32,
        cpu: u32,
        misc: u32,
        stride: u32,
    }

    #[repr(C)]
    struct SubresourceData {
        mem: *const c_void,
        pitch: u32,
        slice_pitch: u32,
    }

    #[repr(C)]
    struct TextureDesc {
        width: u32,
        height: u32,
        mips: u32,
        array: u32,
        format: u32,
        samples: u32,
        quality: u32,
        usage: u32,
        bind: u32,
        cpu: u32,
        misc: u32,
    }

    #[repr(C)]
    struct RasterizerDesc {
        fill: u32,
        cull: u32,
        front_ccw: i32,
        depth_bias: i32,
        bias_clamp: f32,
        slope_bias: f32,
        depth_clip: i32,
        scissor: i32,
        multisample: i32,
        aa_lines: i32,
    }

    #[repr(C)]
    struct StencilOp {
        fail: u32,
        depth_fail: u32,
        pass: u32,
        func: u32,
    }

    #[repr(C)]
    struct DepthDesc {
        depth: i32,
        write: u32,
        func: u32,
        stencil: i32,
        read_mask: u8,
        write_mask: u8,
        front: StencilOp,
        back: StencilOp,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct TargetBlend {
        enable: i32,
        src: u32,
        dst: u32,
        op: u32,
        src_a: u32,
        dst_a: u32,
        op_a: u32,
        mask: u8,
        pad: [u8; 3],
    }

    #[repr(C)]
    struct BlendDesc {
        alpha_to_coverage: i32,
        independent: i32,
        targets: [TargetBlend; 8],
    }

    #[repr(C)]
    struct SamplerDesc {
        filter: u32,
        u: u32,
        v: u32,
        w: u32,
        mip_bias: f32,
        anisotropy: u32,
        compare: u32,
        border: [f32; 4],
        min_lod: f32,
        max_lod: f32,
    }

    #[repr(C)]
    struct Viewport {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        min_depth: f32,
        max_depth: f32,
    }

    #[repr(C)]
    struct ViewDesc {
        format: u32,
        dimension: u32,
        a: u32,
        b: u32,
        c: u32,
        d: u32,
    }

    #[repr(C)]
    struct Mapped {
        data: Ptr,
        row_pitch: u32,
        depth_pitch: u32,
    }

    const BGRA: u32 = 87;
    const R32_TYPELESS: u32 = 39;
    const D32: u32 = 40;
    const R32_FLOAT: u32 = 41;
    const BIND_SRV: u32 = 0x8;
    const BIND_RTV: u32 = 0x20;
    const BIND_DSV: u32 = 0x40;
    const BIND_CONSTANT: u32 = 0x4;
    const MISC_STRUCTURED: u32 = 0x40;
    const MISC_MIPS: u32 = 0x1;

    // the methods used, by their place in each object's table
    const DEV_CREATE_BUFFER: usize = 3;
    const DEV_CREATE_TEXTURE2D: usize = 5;
    const DEV_CREATE_SRV: usize = 7;
    const DEV_CREATE_RTV: usize = 9;
    const DEV_CREATE_DSV: usize = 10;
    const DEV_CREATE_VS: usize = 12;
    const DEV_CREATE_PS: usize = 15;
    const DEV_CREATE_BLEND: usize = 20;
    const DEV_CREATE_DEPTH: usize = 21;
    const DEV_CREATE_RASTER: usize = 22;
    const DEV_CREATE_SAMPLER: usize = 23;
    const DEV_MSAA_LEVELS: usize = 30;
    const CTX_VS_CBS: usize = 7;
    const CTX_PS_SRVS: usize = 8;
    const CTX_PS_SHADER: usize = 9;
    const CTX_PS_SAMPLERS: usize = 10;
    const CTX_VS_SHADER: usize = 11;
    const CTX_DRAW: usize = 13;
    const CTX_MAP: usize = 14;
    const CTX_UNMAP: usize = 15;
    const CTX_PS_CBS: usize = 16;
    const CTX_INPUT_LAYOUT: usize = 17;
    const CTX_TOPOLOGY: usize = 24;
    const CTX_VS_SRVS: usize = 25;
    const CTX_TARGETS: usize = 33;
    const CTX_BLEND: usize = 35;
    const CTX_DEPTH: usize = 36;
    const CTX_RASTER: usize = 43;
    const CTX_VIEWPORTS: usize = 44;
    const CTX_COPY: usize = 47;
    const CTX_UPDATE: usize = 48;
    const CTX_CLEAR_RTV: usize = 50;
    const CTX_CLEAR_DSV: usize = 53;
    const CTX_GENERATE_MIPS: usize = 54;
    const CTX_RESOLVE: usize = 57;

    type CompileFn = unsafe extern "system" fn(*const c_void, usize, *const u8, Ptr, Ptr, *const u8, *const u8, u32, u32, *mut Ptr, *mut Ptr) -> HResult;

    /// One shape to draw.
    pub struct Draw<'a> {
        pub mesh: &'a Mesh,
        pub consts: ItemConsts,
        pub picture: Option<usize>,
        pub casts: bool,
        pub see_through: bool,
    }

    /// The picture being drawn into, and the depth of each pixel.
    struct Target {
        w: u32,
        h: u32,
        colour: Com,
        rtv: Com,
        depth: Com,
        dsv: Com,
        resolved: Com,
        staging: Com,
    }

    pub struct Renderer {
        pub name: String,
        device: Com,
        context: Com,
        compile: CompileFn,
        vs_main: Com,
        ps_main: Com,
        vs_shadow: Com,
        vs_sky: Com,
        ps_sky: Com,
        raster: Com,
        raster_shadow: Com,
        depth_write: Com,
        depth_read: Com,
        depth_off: Com,
        blend_off: Com,
        blend_alpha: Com,
        sampler: Com,
        sampler_shadow: Com,
        cb_frame: Com,
        cb_item: Com,
        shadow_tex: Com,
        shadow_dsv: Com,
        shadow_srv: Com,
        white: (Com, Com),
        samples: u32,
        target: Option<Target>,
        meshes: HashMap<usize, (u64, Com, Com, u32)>,
        pictures: HashMap<usize, (u64, Com, Com)>,
    }

    thread_local! {
        static RENDERER: RefCell<Option<Result<Renderer, String>>> = const { RefCell::new(None) };
    }

    /// Forgets the renderer, so the next picture sets up a new one (after
    /// the graphics card was reset, say).
    pub fn reset() {
        RENDERER.with(|r| *r.borrow_mut() = None);
    }

    pub fn with_renderer<R>(f: impl FnOnce(&mut Renderer) -> Result<R, String>) -> Result<R, String> {
        RENDERER.with(|r| {
            let mut r = r.borrow_mut();
            if r.is_none() {
                *r = Some(Renderer::new());
            }
            match r.as_mut().unwrap() {
                Ok(rd) => f(rd),
                Err(e) => Err(e.clone()),
            }
        })
    }

    fn check(hr: HResult, what: &str) -> Result<(), String> {
        if hr < 0 {
            return Err(format!("the graphics card couldn't {} (error 0x{:08x})", what, hr as u32));
        }
        Ok(())
    }

    impl Renderer {
        fn new() -> Result<Renderer, String> {
            let none = || "3D needs Direct3D 11, which this computer doesn't seem to have".to_string();
            let compile: CompileFn = unsafe { std::mem::transmute(load("d3dcompiler_47.dll", "D3DCompile").ok_or_else(none)?) };
            type CreateFn = unsafe extern "system" fn(Ptr, u32, Ptr, u32, *const u32, u32, u32, *mut Ptr, *mut u32, *mut Ptr) -> HResult;
            let create: CreateFn = unsafe { std::mem::transmute(load("d3d11.dll", "D3D11CreateDevice").ok_or_else(none)?) };
            let levels = [0xb000u32];
            let warp_only = std::env::var("ANTARES_GPU").map(|v| v.eq_ignore_ascii_case("warp")).unwrap_or(false);
            let mut made = None;
            // the graphics card, or else Windows' software renderer (WARP)
            for (driver, name) in [(1u32, ""), (5u32, "Windows' software renderer")] {
                if warp_only && driver == 1 {
                    continue;
                }
                let (mut device, mut context, mut level): (Ptr, Ptr, u32) = (std::ptr::null_mut(), std::ptr::null_mut(), 0);
                let hr = unsafe { create(std::ptr::null_mut(), driver, std::ptr::null_mut(), 0, levels.as_ptr(), 1, 7, &mut device, &mut level, &mut context) };
                if hr >= 0 && !device.is_null() && !context.is_null() {
                    made = Some((Com(device), Com(context), name.to_string()));
                    break;
                }
            }
            let (device, context, mut name) = made.ok_or_else(none)?;
            if name.is_empty() {
                name = "the graphics card (Direct3D 11)".into();
            }
            let null = || Com(std::ptr::null_mut());
            let mut r = Renderer {
                name,
                device,
                context,
                compile,
                vs_main: null(),
                ps_main: null(),
                vs_shadow: null(),
                vs_sky: null(),
                ps_sky: null(),
                raster: null(),
                raster_shadow: null(),
                depth_write: null(),
                depth_read: null(),
                depth_off: null(),
                blend_off: null(),
                blend_alpha: null(),
                sampler: null(),
                sampler_shadow: null(),
                cb_frame: null(),
                cb_item: null(),
                shadow_tex: null(),
                shadow_dsv: null(),
                shadow_srv: null(),
                white: (null(), null()),
                samples: 1,
                target: None,
                meshes: HashMap::new(),
                pictures: HashMap::new(),
            };
            r.vs_main = r.shader("vs_main", "vs_5_0")?;
            r.ps_main = r.shader("ps_main", "ps_5_0")?;
            r.vs_shadow = r.shader("vs_shadow", "vs_5_0")?;
            r.vs_sky = r.shader("vs_sky", "vs_5_0")?;
            r.ps_sky = r.shader("ps_sky", "ps_5_0")?;
            let rd = |cull, slope: f32, ms: i32| RasterizerDesc { fill: 3, cull, front_ccw: 0, depth_bias: 0, bias_clamp: 0.0, slope_bias: slope, depth_clip: 1, scissor: 0, multisample: ms, aa_lines: 0 };
            r.raster = r.raster_state(rd(1, 0.0, 1))?;
            r.raster_shadow = r.raster_state(rd(1, 1.5, 0))?;
            let keep = || StencilOp { fail: 1, depth_fail: 1, pass: 1, func: 8 };
            let dd = |depth, write, func| DepthDesc { depth, write, func, stencil: 0, read_mask: 0xFF, write_mask: 0xFF, front: keep(), back: keep() };
            r.depth_write = r.depth_state(dd(1, 1, 2))?;
            r.depth_read = r.depth_state(dd(1, 0, 2))?;
            r.depth_off = r.depth_state(dd(0, 0, 8))?;
            let tb = |enable| TargetBlend { enable, src: 2, dst: 6, op: 1, src_a: 2, dst_a: 6, op_a: 1, mask: 0xF, pad: [0; 3] };
            r.blend_off = r.blend_state(tb(0))?;
            r.blend_alpha = r.blend_state(tb(1))?;
            r.sampler = r.sampler_state(SamplerDesc { filter: 0x55, u: 1, v: 1, w: 1, mip_bias: 0.0, anisotropy: 8, compare: 1, border: [0.0; 4], min_lod: 0.0, max_lod: f32::MAX })?;
            r.sampler_shadow = r.sampler_state(SamplerDesc { filter: 0x94, u: 4, v: 4, w: 4, mip_bias: 0.0, anisotropy: 1, compare: 4, border: [1.0; 4], min_lod: 0.0, max_lod: f32::MAX })?;
            r.cb_frame = r.buffer(std::mem::size_of::<FrameConsts>() as u32, BIND_CONSTANT, 0, 0, None)?;
            r.cb_item = r.buffer(std::mem::size_of::<ItemConsts>() as u32, BIND_CONSTANT, 0, 0, None)?;
            // the sun's shadow map: how far the sun sees, from its side
            let st = r.texture(TextureDesc { width: SHADOW_SIZE, height: SHADOW_SIZE, mips: 1, array: 1, format: R32_TYPELESS, samples: 1, quality: 0, usage: 0, bind: BIND_DSV | BIND_SRV, cpu: 0, misc: 0 }, None)?;
            r.shadow_dsv = r.view(DEV_CREATE_DSV, &st, Some(&ViewDesc { format: D32, dimension: 3, a: 0, b: 0, c: 0, d: 0 }))?;
            r.shadow_srv = r.view(DEV_CREATE_SRV, &st, Some(&ViewDesc { format: R32_FLOAT, dimension: 4, a: 0, b: 1, c: 0, d: 0 }))?;
            r.shadow_tex = st;
            let white = r.texture(TextureDesc { width: 1, height: 1, mips: 1, array: 1, format: BGRA, samples: 1, quality: 0, usage: 0, bind: BIND_SRV, cpu: 0, misc: 0 }, Some((&[0xFFFF_FFFFu32], 4)))?;
            let wsrv = r.view(DEV_CREATE_SRV, &white, None)?;
            r.white = (white, wsrv);
            // smooth edges: 4 samples a pixel, if the card can
            let levels = |format: u32| {
                let f: unsafe extern "system" fn(Ptr, u32, u32, *mut u32) -> HResult = unsafe { method(r.device.0, DEV_MSAA_LEVELS) };
                let mut q = 0u32;
                let hr = unsafe { f(r.device.0, format, 4, &mut q) };
                hr >= 0 && q > 0
            };
            r.samples = if levels(BGRA) && levels(D32) { 4 } else { 1 };
            Ok(r)
        }

        fn shader(&self, entry: &str, target: &str) -> Result<Com, String> {
            let src = super::SHADERS;
            let cache = shader_cache_file(src, entry, target);
            let bytes = match cache.as_ref().and_then(|f| std::fs::read(f).ok()) {
                Some(b) if !b.is_empty() => b,
                _ => {
                    let b = self.compile(src, entry, target)?;
                    if let Some(f) = &cache {
                        if let Some(dir) = f.parent() {
                            let _ = std::fs::create_dir_all(dir);
                        }
                        let tmp = f.with_extension(format!("tmp{}", std::process::id()));
                        if std::fs::write(&tmp, &b).is_ok() && std::fs::rename(&tmp, f).is_err() {
                            let _ = std::fs::remove_file(&tmp);
                        }
                    }
                    b
                }
            };
            let index = if target.starts_with("vs") { DEV_CREATE_VS } else { DEV_CREATE_PS };
            let create: unsafe extern "system" fn(Ptr, *const c_void, usize, Ptr, *mut Ptr) -> HResult = unsafe { method(self.device.0, index) };
            let mut s: Ptr = std::ptr::null_mut();
            if unsafe { create(self.device.0, bytes.as_ptr().cast(), bytes.len(), std::ptr::null_mut(), &mut s) } < 0 {
                if let Some(f) = &cache {
                    let _ = std::fs::remove_file(f);
                }
                let fresh = self.compile(src, entry, target)?;
                check(unsafe { create(self.device.0, fresh.as_ptr().cast(), fresh.len(), std::ptr::null_mut(), &mut s) }, "load a 3D program")?;
            }
            Ok(Com(s))
        }

        fn compile(&self, src: &str, entry: &str, target: &str) -> Result<Vec<u8>, String> {
            let (mut code, mut errors): (Ptr, Ptr) = (std::ptr::null_mut(), std::ptr::null_mut());
            let entry_c: Vec<u8> = entry.bytes().chain([0]).collect();
            let target_c: Vec<u8> = target.bytes().chain([0]).collect();
            let hr = unsafe { (self.compile)(src.as_ptr().cast(), src.len(), c"antares3d".as_ptr().cast(), std::ptr::null_mut(), std::ptr::null_mut(), entry_c.as_ptr(), target_c.as_ptr(), 1 << 15, 0, &mut code, &mut errors) };
            let (code, errors) = (Com(code), Com(errors));
            if hr < 0 || code.0.is_null() {
                let msg = if errors.0.is_null() {
                    String::new()
                } else {
                    unsafe {
                        let p: unsafe extern "system" fn(Ptr) -> *const u8 = method(errors.0, 3);
                        let n: unsafe extern "system" fn(Ptr) -> usize = method(errors.0, 4);
                        String::from_utf8_lossy(std::slice::from_raw_parts(p(errors.0), n(errors.0))).into_owned()
                    }
                };
                return Err(format!("a 3D program didn't compile: {}", msg.trim()));
            }
            unsafe {
                let p: unsafe extern "system" fn(Ptr) -> *const u8 = method(code.0, 3);
                let n: unsafe extern "system" fn(Ptr) -> usize = method(code.0, 4);
                Ok(std::slice::from_raw_parts(p(code.0), n(code.0)).to_vec())
            }
        }

        fn make<D>(&self, index: usize, desc: &D, what: &str) -> Result<Com, String> {
            let f: unsafe extern "system" fn(Ptr, *const D, *mut Ptr) -> HResult = unsafe { method(self.device.0, index) };
            let mut out: Ptr = std::ptr::null_mut();
            check(unsafe { f(self.device.0, desc, &mut out) }, what)?;
            Ok(Com(out))
        }

        fn raster_state(&self, d: RasterizerDesc) -> Result<Com, String> {
            self.make(DEV_CREATE_RASTER, &d, "set up drawing")
        }

        fn depth_state(&self, d: DepthDesc) -> Result<Com, String> {
            self.make(DEV_CREATE_DEPTH, &d, "set up depth")
        }

        fn blend_state(&self, t: TargetBlend) -> Result<Com, String> {
            self.make(DEV_CREATE_BLEND, &BlendDesc { alpha_to_coverage: 0, independent: 0, targets: [t; 8] }, "set up blending")
        }

        fn sampler_state(&self, d: SamplerDesc) -> Result<Com, String> {
            self.make(DEV_CREATE_SAMPLER, &d, "set up pictures")
        }

        fn buffer(&self, size: u32, bind: u32, misc: u32, stride: u32, init: Option<&[u8]>) -> Result<Com, String> {
            let f: unsafe extern "system" fn(Ptr, *const BufferDesc, *const SubresourceData, *mut Ptr) -> HResult = unsafe { method(self.device.0, DEV_CREATE_BUFFER) };
            let data = init.map(|b| SubresourceData { mem: b.as_ptr().cast(), pitch: 0, slice_pitch: 0 });
            let mut out: Ptr = std::ptr::null_mut();
            let desc = BufferDesc { byte_width: size, usage: 0, bind, cpu: 0, misc, stride };
            check(unsafe { f(self.device.0, &desc, data.as_ref().map_or(std::ptr::null(), |d| d as *const _), &mut out) }, "make room for a shape")?;
            Ok(Com(out))
        }

        fn texture(&self, d: TextureDesc, init: Option<(&[u32], u32)>) -> Result<Com, String> {
            let f: unsafe extern "system" fn(Ptr, *const TextureDesc, *const SubresourceData, *mut Ptr) -> HResult = unsafe { method(self.device.0, DEV_CREATE_TEXTURE2D) };
            let data = init.map(|(px, pitch)| SubresourceData { mem: px.as_ptr().cast(), pitch, slice_pitch: 0 });
            let mut out: Ptr = std::ptr::null_mut();
            check(unsafe { f(self.device.0, &d, data.as_ref().map_or(std::ptr::null(), |d| d as *const _), &mut out) }, "make room for a picture")?;
            Ok(Com(out))
        }

        fn view(&self, index: usize, res: &Com, desc: Option<&ViewDesc>) -> Result<Com, String> {
            let f: unsafe extern "system" fn(Ptr, Ptr, *const ViewDesc, *mut Ptr) -> HResult = unsafe { method(self.device.0, index) };
            let mut out: Ptr = std::ptr::null_mut();
            check(unsafe { f(self.device.0, res.0, desc.map_or(std::ptr::null(), |d| d as *const _), &mut out) }, "set up a picture")?;
            Ok(Com(out))
        }

        fn ensure_target(&mut self, w: u32, h: u32) -> Result<(), String> {
            if self.target.as_ref().is_some_and(|t| t.w == w && t.h == h) {
                return Ok(());
            }
            self.target = None;
            let td = |format, samples, bind, usage, cpu| TextureDesc { width: w, height: h, mips: 1, array: 1, format, samples, quality: 0, usage, bind, cpu, misc: 0 };
            let colour = self.texture(td(BGRA, self.samples, BIND_RTV, 0, 0), None)?;
            let rtv = self.view(DEV_CREATE_RTV, &colour, None)?;
            let depth = self.texture(td(D32, self.samples, BIND_DSV, 0, 0), None)?;
            let dsv = self.view(DEV_CREATE_DSV, &depth, None)?;
            let resolved = self.texture(td(BGRA, 1, BIND_SRV, 0, 0), None)?;
            let staging = self.texture(td(BGRA, 1, 0, 3, 0x20000), None)?;
            self.target = Some(Target { w, h, colour, rtv, depth, dsv, resolved, staging });
            Ok(())
        }

        fn mesh_srv(&mut self, m: &Mesh) -> Result<(*mut c_void, u32), String> {
            let n = (m.verts.len() / 8) as u32;
            let bytes: &[u8] = unsafe { std::slice::from_raw_parts(m.verts.as_ptr().cast(), m.verts.len() * 4) };
            match self.meshes.get_mut(&m.id) {
                Some(e) if e.0 == m.version => {}
                // corners moved, same number of them: overwrite in place
                Some(e) if e.3 == n && n > 0 => {
                    let c = self.context.0;
                    unsafe {
                        let update: unsafe extern "system" fn(Ptr, Ptr, u32, *const c_void, *const c_void, u32, u32) = method(c, CTX_UPDATE);
                        update(c, e.1.0, 0, std::ptr::null(), bytes.as_ptr().cast(), 0, 0);
                    }
                    e.0 = m.version;
                }
                _ => {
                    let buf = self.buffer((bytes.len() as u32).max(32), BIND_SRV, MISC_STRUCTURED, 32, if bytes.is_empty() { None } else { Some(bytes) })?;
                    let srv = self.view(DEV_CREATE_SRV, &buf, None)?;
                    self.meshes.insert(m.id, (m.version, buf, srv, n));
                }
            }
            let e = &self.meshes[&m.id];
            Ok((e.2.0, e.3))
        }

        /// A picture on the graphics card, sent again whenever it has been
        /// drawn on since.
        fn picture_srv(&mut self, id: usize) -> Result<*mut c_void, String> {
            let (version, w, h) = crate::gfx::picture_version(id);
            if self.pictures.get(&id).is_none_or(|p| p.0 != version) {
                let px = crate::gfx::picture_pixels(id);
                if w == 0 || h == 0 {
                    return Ok(self.white.1.0);
                }
                let tex = self.texture(TextureDesc { width: w as u32, height: h as u32, mips: 0, array: 1, format: BGRA, samples: 1, quality: 0, usage: 0, bind: BIND_SRV | BIND_RTV, cpu: 0, misc: MISC_MIPS }, None)?;
                let srv = self.view(DEV_CREATE_SRV, &tex, None)?;
                let c = self.context.0;
                unsafe {
                    let update: unsafe extern "system" fn(Ptr, Ptr, u32, *const c_void, *const c_void, u32, u32) = method(c, CTX_UPDATE);
                    update(c, tex.0, 0, std::ptr::null(), px.as_ptr().cast(), (w * 4) as u32, 0);
                    let mips: unsafe extern "system" fn(Ptr, Ptr) = method(c, CTX_GENERATE_MIPS);
                    mips(c, srv.0);
                }
                self.pictures.insert(id, (version, tex, srv));
            }
            Ok(self.pictures[&id].2.0)
        }

        fn update<T>(&self, buf: &Com, data: &T) {
            let c = self.context.0;
            unsafe {
                let update: unsafe extern "system" fn(Ptr, Ptr, u32, *const c_void, *const c_void, u32, u32) = method(c, CTX_UPDATE);
                update(c, buf.0, 0, std::ptr::null(), (data as *const T).cast(), 0, 0);
            }
        }

        pub fn render(&mut self, w: usize, h: usize, frame: &FrameConsts, draws: &[Draw], sky: bool, shadows: bool) -> Result<Vec<u32>, String> {
            let (w32, h32) = (w as u32, h as u32);
            self.ensure_target(w32, h32)?;
            // everything the shapes need on the card first
            let mut prepared = Vec::with_capacity(draws.len());
            for d in draws {
                let (srv, n) = self.mesh_srv(d.mesh)?;
                let pic = match d.picture {
                    Some(id) => self.picture_srv(id)?,
                    None => self.white.1.0,
                };
                prepared.push((srv, n, pic));
            }
            self.update(&self.cb_frame, frame);
            let c = self.context.0;
            let t = self.target.as_ref().unwrap();
            unsafe {
                let set_targets: unsafe extern "system" fn(Ptr, u32, *const Ptr, Ptr) = method(c, CTX_TARGETS);
                let clear_rtv: unsafe extern "system" fn(Ptr, Ptr, *const [f32; 4]) = method(c, CTX_CLEAR_RTV);
                let clear_dsv: unsafe extern "system" fn(Ptr, Ptr, u32, f32, u8) = method(c, CTX_CLEAR_DSV);
                let viewports: unsafe extern "system" fn(Ptr, u32, *const Viewport) = method(c, CTX_VIEWPORTS);
                let raster: unsafe extern "system" fn(Ptr, Ptr) = method(c, CTX_RASTER);
                let blend: unsafe extern "system" fn(Ptr, Ptr, *const [f32; 4], u32) = method(c, CTX_BLEND);
                let depth: unsafe extern "system" fn(Ptr, Ptr, u32) = method(c, CTX_DEPTH);
                let topology: unsafe extern "system" fn(Ptr, u32) = method(c, CTX_TOPOLOGY);
                let layout: unsafe extern "system" fn(Ptr, Ptr) = method(c, CTX_INPUT_LAYOUT);
                let vs: unsafe extern "system" fn(Ptr, Ptr, *const Ptr, u32) = method(c, CTX_VS_SHADER);
                let ps: unsafe extern "system" fn(Ptr, Ptr, *const Ptr, u32) = method(c, CTX_PS_SHADER);
                let vs_srvs: unsafe extern "system" fn(Ptr, u32, u32, *const Ptr) = method(c, CTX_VS_SRVS);
                let ps_srvs: unsafe extern "system" fn(Ptr, u32, u32, *const Ptr) = method(c, CTX_PS_SRVS);
                let vs_cbs: unsafe extern "system" fn(Ptr, u32, u32, *const Ptr) = method(c, CTX_VS_CBS);
                let ps_cbs: unsafe extern "system" fn(Ptr, u32, u32, *const Ptr) = method(c, CTX_PS_CBS);
                let samplers: unsafe extern "system" fn(Ptr, u32, u32, *const Ptr) = method(c, CTX_PS_SAMPLERS);
                let draw: unsafe extern "system" fn(Ptr, u32, u32) = method(c, CTX_DRAW);

                topology(c, 4); // triangle lists
                layout(c, std::ptr::null_mut());
                let cbs = [self.cb_frame.0, self.cb_item.0];
                vs_cbs(c, 0, 2, cbs.as_ptr());
                ps_cbs(c, 0, 2, cbs.as_ptr());
                let none: [Ptr; 3] = [std::ptr::null_mut(); 3];
                ps_srvs(c, 0, 3, none.as_ptr());

                // 1. the sun's view, for shadows
                if shadows {
                    set_targets(c, 0, std::ptr::null(), self.shadow_dsv.0);
                    clear_dsv(c, self.shadow_dsv.0, 1, 1.0, 0);
                    viewports(c, 1, &Viewport { x: 0.0, y: 0.0, w: SHADOW_SIZE as f32, h: SHADOW_SIZE as f32, min_depth: 0.0, max_depth: 1.0 });
                    raster(c, self.raster_shadow.0);
                    depth(c, self.depth_write.0, 0);
                    blend(c, self.blend_off.0, std::ptr::null(), 0xFFFF_FFFF);
                    vs(c, self.vs_shadow.0, std::ptr::null(), 0);
                    ps(c, std::ptr::null_mut(), std::ptr::null(), 0);
                    for (d, p) in draws.iter().zip(&prepared) {
                        if d.casts && p.1 > 0 {
                            self.update(&self.cb_item, &d.consts);
                            vs_srvs(c, 0, 1, &p.0);
                            draw(c, p.1, 0);
                        }
                    }
                }

                // 2. the picture
                set_targets(c, 1, &t.rtv.0, t.dsv.0);
                clear_rtv(c, t.rtv.0, &[0.0; 4]);
                clear_dsv(c, t.dsv.0, 1, 1.0, 0);
                viewports(c, 1, &Viewport { x: 0.0, y: 0.0, w: w as f32, h: h as f32, min_depth: 0.0, max_depth: 1.0 });
                raster(c, self.raster.0);
                let samps = [self.sampler.0, self.sampler_shadow.0];
                samplers(c, 0, 2, samps.as_ptr());
                if sky {
                    depth(c, self.depth_off.0, 0);
                    blend(c, self.blend_off.0, std::ptr::null(), 0xFFFF_FFFF);
                    vs(c, self.vs_sky.0, std::ptr::null(), 0);
                    ps(c, self.ps_sky.0, std::ptr::null(), 0);
                    draw(c, 3, 0);
                }
                vs(c, self.vs_main.0, std::ptr::null(), 0);
                ps(c, self.ps_main.0, std::ptr::null(), 0);
                let mut see_through = false;
                depth(c, self.depth_write.0, 0);
                blend(c, self.blend_off.0, std::ptr::null(), 0xFFFF_FFFF);
                for (d, p) in draws.iter().zip(&prepared) {
                    if p.1 == 0 {
                        continue;
                    }
                    if d.see_through && !see_through {
                        see_through = true;
                        depth(c, self.depth_read.0, 0);
                        blend(c, self.blend_alpha.0, std::ptr::null(), 0xFFFF_FFFF);
                    }
                    self.update(&self.cb_item, &d.consts);
                    vs_srvs(c, 0, 1, &p.0);
                    let srvs = [p.2, self.shadow_srv.0];
                    ps_srvs(c, 1, 2, srvs.as_ptr());
                    draw(c, p.1, 0);
                }
                ps_srvs(c, 0, 3, none.as_ptr());
                set_targets(c, 0, std::ptr::null(), std::ptr::null_mut());

                // 3. back to the program
                if self.samples > 1 {
                    let resolve: unsafe extern "system" fn(Ptr, Ptr, u32, Ptr, u32, u32) = method(c, CTX_RESOLVE);
                    resolve(c, t.resolved.0, 0, t.colour.0, 0, BGRA);
                } else {
                    let copy: unsafe extern "system" fn(Ptr, Ptr, Ptr) = method(c, CTX_COPY);
                    copy(c, t.resolved.0, t.colour.0);
                }
                let copy: unsafe extern "system" fn(Ptr, Ptr, Ptr) = method(c, CTX_COPY);
                copy(c, t.staging.0, t.resolved.0);
                let map: unsafe extern "system" fn(Ptr, Ptr, u32, u32, u32, *mut Mapped) -> HResult = method(c, CTX_MAP);
                let unmap: unsafe extern "system" fn(Ptr, Ptr, u32) = method(c, CTX_UNMAP);
                let mut m: Mapped = std::mem::zeroed();
                check(map(c, t.staging.0, 0, 1, 0, &mut m), "hand back the picture")?;
                let mut out = Vec::with_capacity(w * h);
                for y in 0..h {
                    let row = (m.data as *const u8).add(y * m.row_pitch as usize) as *const u32;
                    out.extend_from_slice(std::slice::from_raw_parts(row, w));
                }
                unmap(c, t.staging.0, 0);
                let _ = &t.depth;
                Ok(out)
            }
        }
    }

    /// Where a compiled 3D program is kept, like the GPU's other programs.
    fn shader_cache_file(src: &str, entry: &str, target: &str) -> Option<std::path::PathBuf> {
        let base = std::env::var_os("LOCALAPPDATA").filter(|v| !v.is_empty()).map(std::path::PathBuf::from)?;
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in src.bytes().chain(entry.bytes()).chain(target.bytes()).chain(env!("CARGO_PKG_VERSION").bytes()) {
            h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
        Some(base.join("antares").join("shaders").join(format!("{:016x}.cso", h)))
    }
}

/// The shaders: programs the graphics card runs for each corner and each
/// pixel. Light is worked out in linear units, then mapped to the screen
/// with a filmic curve (ACES), which keeps bright light from clipping.
#[cfg_attr(not(windows), allow(dead_code))]
const SHADERS: &str = r#"
cbuffer Frame : register(b0) {
    row_major float4x4 viewProj;
    row_major float4x4 lightVP;
    row_major float4x4 invViewProj;
    float4 camPos;
    float4 sunDir;      // xyz: towards the sun; w: 1 if there is one
    float4 sunCol;      // w: 1 if it casts shadows
    float4 skyTop;
    float4 skyHor;
    float4 skyGnd;
    float4 fogCol;      // w: 1 if there is fog
    float4 params;      // fog near, fog far, seconds, number of lamps
    float4 shadowInfo;  // size of a shadow-map pixel in the world, and on the map; a pixel's width per unit away
    float4 lampPos[8];  // w: reach
    float4 lampCol[8];
};
cbuffer Item : register(b1) {
    row_major float4x4 world;
    row_major float4x4 nworld;
    float4 albedo;      // colour in light units, and alpha
    float4 mat;         // shine, metal, glow, tiles
    float4 flags;       // has a picture, is see-through, ripples
};
struct Vert { float3 p; float3 n; float2 uv; };
StructuredBuffer<Vert> verts : register(t0);
Texture2D tex : register(t1);
Texture2D<float> shadowMap : register(t2);
SamplerState samp : register(s0);
SamplerComparisonState shadowSamp : register(s1);

struct PSIn {
    float4 pos : SV_Position;
    float3 wp : TEXCOORD0;
    float3 n : TEXCOORD1;
    float2 uv : TEXCOORD2;
};

PSIn vs_main(uint id : SV_VertexID) {
    Vert v = verts[id];
    PSIn o;
    float4 w = mul(float4(v.p, 1), world);
    o.wp = w.xyz;
    o.pos = mul(w, viewProj);
    o.n = mul(float4(v.n, 0), nworld).xyz;
    o.uv = v.uv * mat.w;
    return o;
}

float4 vs_shadow(uint id : SV_VertexID) : SV_Position {
    Vert v = verts[id];
    return mul(mul(float4(v.p, 1), world), lightVP);
}

float3 skyColour(float3 d) {
    if (d.y >= 0) return lerp(skyHor.rgb, skyTop.rgb, pow(saturate(d.y), 0.4));
    return lerp(skyHor.rgb, skyGnd.rgb, pow(saturate(-d.y), 0.35));
}

float3 skyLight(float3 n) {
    if (n.y >= 0) return lerp(skyHor.rgb, skyTop.rgb, n.y);
    return lerp(skyHor.rgb, skyGnd.rgb, -n.y);
}

float3 gamma(float3 c) {
    return pow(saturate(c), 1.0 / 2.2);
}

float aces(float x) {
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

// The filmic curve applied to the brightest part of each colour, so colours
// keep their hue and strength, and only the very brightest light turns a
// little towards white.
float3 finish(float3 c) {
    c = max(c, 0) * 0.6;
    float m = max(c.r, max(c.g, c.b));
    float3 r = c * (aces(m) / max(m, 1e-5));
    r = lerp(r, aces(m), saturate((m - 2) * 0.1));
    return gamma(r);
}

float sunShadow(float3 wp, float3 n) {
    float4 lp = mul(float4(wp + n * shadowInfo.x * 1.5, 1), lightVP);
    float2 uv = float2(lp.x * 0.5 + 0.5, 0.5 - lp.y * 0.5);
    if (uv.x < 0 || uv.x > 1 || uv.y < 0 || uv.y > 1 || lp.z > 1) return 1;
    float s = 0;
    [unroll] for (int y = -1; y <= 1; y++)
        [unroll] for (int x = -1; x <= 1; x++)
            s += shadowMap.SampleCmpLevelZero(shadowSamp, uv + float2(x, y) * shadowInfo.y * 1.6, lp.z - 0.0008);
    return s / 9.0;
}

// Small waves too fine for the shape's corners, worked out for each pixel:
// fourteen trains of ripples, from 2.5 m long down to a few centimetres,
// spread round one direction like wind-blown water, each moving at the
// speed water waves of its length travel. Ripples too small for a pixel
// (counting how stretched pixels are, seen at a glancing angle) fade out
// so the distance doesn't shimmer, and what they would have added comes
// back as roughness: far water looks smoother, with a wider path of glints.
float3 rippled(float3 n, float3 wp, float3 v, float amount, inout float lost) {
    float2 p = wp.xz;
    float glance = max(abs(dot(n, v)), 0.3);
    float pixel = max(length(camPos.xyz - wp) * shadowInfo.z / glance, 1e-4);
    float2 g = 0;
    [unroll] for (int k = 0; k < 14; k++) {
        float L = 2.5 * pow(0.76, k);
        float kk = 6.2832 / L;
        float ang = 0.35 + sin(k * 12.9898) * 1.2;
        float2 d = float2(cos(ang), sin(ang));
        float fade = saturate(L / (pixel * 6) - 1);
        float ph = dot(d, p) * kk - sqrt(9.81 * kk) * params.z + k * 4.37;
        float slope = 0.075 * amount;
        g += d * (slope * cos(ph) * fade);
        lost += slope * slope * 0.5 * (1 - fade * fade);
    }
    return normalize(n + float3(-g.x, 0, -g.y) * n.y);
}

// light arriving from direction l, reflected towards v
float3 shade(float3 n, float3 v, float3 l, float nv, float a, float3 f0, float3 diff) {
    float nl = saturate(dot(n, l));
    if (nl <= 0) return 0;
    float3 h = normalize(l + v);
    float nh = saturate(dot(n, h));
    float vh = saturate(dot(v, h));
    float3 F = f0 + (1 - f0) * pow(1 - vh, 5);
    float a2 = a * a;
    float dd = nh * nh * (a2 - 1) + 1;
    float D = a2 / (3.14159 * dd * dd);
    float k = a * 0.5;
    float G = (nl / (nl * (1 - k) + k)) * (nv / (nv * (1 - k) + k));
    float3 spec = D * F * G / (4 * nl * nv + 1e-4);
    return (diff * (1 - F) + spec * 3.14159) * nl;
}

float4 ps_main(PSIn i) : SV_Target {
    float4 base = albedo;
    if (flags.x > 0) {
        float4 t = tex.Sample(samp, i.uv);
        float3 c = t.a > 0.001 ? t.rgb / t.a : 0;
        base.rgb *= pow(c, 2.2);
        base.a *= t.a;
    }
    if (flags.y < 0.5) {
        clip(base.a - 0.5);
        base.a = 1;
    }
    float3 v = normalize(camPos.xyz - i.wp);
    float3 n = normalize(i.n);
    float3 g = normalize(cross(ddx(i.wp), ddy(i.wp)));
    if (dot(g, v) < 0) g = -g;
    if (dot(n, g) < 0) n = -n;
    float lost = 0;
    if (flags.z > 0) n = rippled(n, i.wp, v, flags.z, lost);
    float nv = saturate(dot(n, v)) + 1e-4;
    float rough = clamp(1 - mat.x * 0.95, 0.05, 1);
    float a = sqrt(rough * rough * rough * rough + lost);
    rough = sqrt(a);
    float metal = mat.y;
    float3 f0 = lerp(0.04, base.rgb, metal);
    float3 diff = base.rgb * (1 - metal);
    float3 col = 0;
    if (sunDir.w > 0) {
        float3 l = sunDir.xyz;
        if (dot(n, l) > 0) {
            float sh = sunCol.w > 0 ? sunShadow(i.wp, n) : 1;
            col += shade(n, v, l, nv, a, f0, diff) * sunCol.rgb * sh;
        }
    }
    int lamps = (int)params.w;
    for (int j = 0; j < lamps; j++) {
        float3 L = lampPos[j].xyz - i.wp;
        float d = length(L);
        float r = d / lampPos[j].w;
        float fall = saturate(1 - r * r * r * r);
        fall = fall * fall / (d * d + 1);
        col += shade(n, v, L / max(d, 1e-4), nv, a, f0, diff) * lampCol[j].rgb * fall;
    }
    float3 Fa = f0 + (max(1 - rough, f0) - f0) * pow(1 - nv, 5);
    col += diff * skyLight(n) * 0.3 * (1 - Fa * 0.5);
    col += skyColour(reflect(-v, n)) * Fa * (1 - rough * 0.85);
    col += base.rgb * mat.z * 2.5;
    // light through the thin tops of water waves, lit from behind
    if (flags.z > 0 && sunDir.w > 0) col += diff * sunCol.rgb * 0.6 * pow(saturate(dot(-v, sunDir.xyz) * 0.5 + 0.5), 4) * saturate(n.y);
    float3 c = finish(col);
    // fog and sky are shown as their own colours, so they meet
    if (fogCol.w > 0) {
        float f = saturate((length(camPos.xyz - i.wp) - params.x) / max(params.y - params.x, 1e-3));
        c = lerp(c, gamma(fogCol.rgb), f * f * (3 - 2 * f));
    }
    return float4(c * base.a, base.a);
}

struct SkyOut { float4 pos : SV_Position; float2 ndc : TEXCOORD0; };

SkyOut vs_sky(uint id : SV_VertexID) {
    float2 p = float2((id << 1) & 2, id & 2) * 2 - 1;
    SkyOut o;
    o.pos = float4(p, 1, 1);
    o.ndc = p;
    return o;
}

float4 ps_sky(SkyOut i) : SV_Target {
    float4 w = mul(float4(i.ndc, 1, 1), invViewProj);
    float3 d = normalize(w.xyz / w.w - camPos.xyz);
    float3 c = skyColour(d);
    if (sunDir.w > 0) {
        float s = saturate(dot(d, sunDir.xyz));
        c += sunCol.rgb * (pow(s, 6000) * 10 + pow(s, 300) * 0.25 + pow(s, 12) * 0.12);
    }
    if (fogCol.w > 0) c = lerp(c, fogCol.rgb, saturate(1 - abs(d.y) * 4) * 0.8);
    return float4(gamma(c), 1);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrices() {
        let m = mmul(&translate([1.0, 2.0, 3.0]), &scaling([2.0, 2.0, 2.0]));
        assert_eq!(apply([1.0, 1.0, 1.0], &m), [4.0, 6.0, 8.0, 1.0]);
        let r = turning([0.0, 90.0, 0.0]);
        let p = apply([1.0, 0.0, 0.0], &r);
        assert!((p[0]).abs() < 1e-6 && (p[2] + 1.0).abs() < 1e-6, "{:?}", p);
        let i = invert(&mmul(&r, &m));
        let back = apply(apply([0.3, -2.0, 5.0], &mmul(&r, &m))[..3].try_into().unwrap(), &i);
        assert!((back[0] - 0.3).abs() < 1e-4 && (back[1] + 2.0).abs() < 1e-4 && (back[2] - 5.0).abs() < 1e-4);
        // the camera looks down -z; a point in front lands in the middle
        let vp = mmul(&look_at([0.0, 0.0, 5.0], [0.0; 3], [0.0, 1.0, 0.0]), &perspective(60.0, 1.0, 0.1, 100.0));
        let c = apply([0.0, 0.0, 0.0], &vp);
        assert!(c[0].abs() < 1e-6 && c[1].abs() < 1e-6 && c[2] / c[3] > 0.0 && c[2] / c[3] < 1.0);
    }

    #[test]
    fn shapes_are_closed_and_sized() {
        for (i, v) in [cube_mesh(), sphere_mesh(), round_mesh(1.0), round_mesh(0.0), plane_mesh()].into_iter().enumerate() {
            assert_eq!(v.len() % 24, 0, "shape {}", i);
            let m = make_mesh(i, v);
            assert!(m.size.iter().all(|s| *s <= 2.0001), "shape {} is {:?}", i, m.size);
        }
    }

    #[test]
    fn obj_files() {
        let src = "# a square\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 1\nf 1/1 2/1 3/2 4/2\n";
        let v = read_obj(src).unwrap();
        assert_eq!(v.len(), 2 * 3 * 8); // a quad: two triangles
        assert!((v[5] - 1.0).abs() < 1e-6 || (v[5] + 1.0).abs() < 1e-6); // normal along z
        assert!(read_obj("v 0 0 0\nf 1 2 3\n").unwrap_err().contains("isn't there"));
        assert!(read_obj("v 0 0\n").is_err());
        assert!(read_obj("v -1 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n").is_ok());
    }
}
