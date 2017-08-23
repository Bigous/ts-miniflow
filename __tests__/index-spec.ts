import * as index from '../src/index';

test('Should have Node available', () => {
  expect(index.Node).toBeTruthy();
});

test('Should have Input available', () => {
  expect(index.Input).toBeTruthy();
});

test('Should have Linear available', () => {
  expect(index.Linear).toBeTruthy();
});

test('Should have Sigmoid available', () => {
  expect(index.Sigmoid).toBeTruthy();
});

test('Should have MSE available', () => {
  expect(index.MSE).toBeTruthy();
});
